"""Bounded offline runtime soak with real SQLite, optional HTTP monitoring and PDF workers."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import httpx

from sibyl.async_cache import CapacityError
from sibyl.config import Config
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.storage import ArtifactStore


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def descriptor_count() -> int | None:
    for path in (Path("/proc/self/fd"), Path("/dev/fd")):
        if path.is_dir():
            return len(list(path.iterdir()))
    return None


def child_processes() -> set[int] | None:
    if os.name != "posix":
        return None
    with subprocess.Popen(
        ["ps", "-axo", "pid=,ppid="], stdout=subprocess.PIPE, text=True
    ) as probe:
        output, _ = probe.communicate(timeout=5)
        require(probe.returncode == 0, "Could not sample child processes")
        return {
            int(pid)
            for pid, parent in (line.split() for line in output.splitlines())
            if int(parent) == os.getpid() and int(pid) != probe.pid
        }


async def check(rounds: int, monitor: bool = False, pdf: bool = False) -> dict:
    started = time.monotonic()
    token = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="sibyl-soak-") as directory:
        root = Path(directory) / "runtime"
        runtime = SibylRuntime(
            RuntimePolicy(
                output_root=root, tool_timeout=0.2, monitor_port=0 if monitor else None
            ),
            Config(),
        )
        jobs = runtime.research_jobs
        baseline_tasks = asyncio.all_tasks()
        baseline_children = child_processes()
        baseline_fds = descriptor_count()
        measurements = []
        pdf_calls = 0
        try:
            with mock.patch.dict(os.environ, {"SIBYL_MONITOR_TOKEN": token}):
                await runtime.start()
            async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                monitor_url = (
                    f"http://127.0.0.1:{runtime.monitor.port}/status" if monitor else ""
                )

                async def cycle(index: int) -> None:
                    nonlocal pdf_calls
                    mode = index % 3
                    entered = asyncio.Event()
                    release = asyncio.Event()
                    count = 0

                    async def upstream(*args, **kwargs):
                        nonlocal count
                        count += 1
                        if count == 2:
                            entered.set()
                        await release.wait()
                        raise httpx.ReadError("Injected upstream failure")

                    created = [
                        jobs.start("Soak question", [], "host", 1, "en", 3)
                        for _ in range(3)
                    ]
                    tasks = []
                    with mock.patch(
                        "sibyl.research_jobs.gather_source_bundle", upstream
                    ):
                        try:
                            for job in created[:2]:
                                tasks.append(
                                    asyncio.create_task(
                                        runtime.execute(
                                            "continue_research",
                                            lambda job=job: jobs.continue_job(
                                                job["job_id"], "focused query"
                                            ),
                                        )
                                    )
                                )
                            await asyncio.wait_for(entered.wait(), 2)
                            try:
                                await jobs.continue_job(created[2]["job_id"], "query")
                            except CapacityError:
                                pass
                            else:
                                raise RuntimeError(
                                    "Worker capacity limit was not enforced"
                                )
                            if mode == 0:
                                release.set()
                            elif mode == 1:
                                for job in created[:2]:
                                    await jobs.cancel(job["job_id"])
                            results = await asyncio.gather(
                                *tasks, return_exceptions=True
                            )
                            require(
                                all(
                                    isinstance(result, asyncio.CancelledError)
                                    for result in results
                                )
                                if mode == 1
                                else all(
                                    isinstance(result, dict) for result in results
                                ),
                                "Unexpected failure/cancellation outcome",
                            )
                        finally:
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                    for job in created[:2]:
                        saved = jobs.get(job["job_id"])
                        require(
                            saved["remaining_steps"] == 2,
                            "Failed retrieval refunded its budget",
                        )
                        require(
                            saved["attempts"][0]["status"] == "failed",
                            "Failed attempt lost",
                        )
                        require(not saved["evidence"], "Failure fabricated evidence")
                    require(
                        jobs.get(created[2]["job_id"])["remaining_steps"] == 3,
                        "Rejected job consumed budget",
                    )
                    for job in created:
                        await jobs.cancel(job["job_id"], delete=True)
                    require(
                        not jobs.tasks and not runtime._tasks,
                        "Workers leaked after cycle",
                    )
                    require(not jobs.store.list_jobs(), "Deleted jobs accumulated")
                    artifact = ArtifactStore(root).write(
                        b"bounded soak artifact", "report", ".md"
                    )
                    artifact.unlink()
                    if monitor:
                        response = await client.get(
                            monitor_url, headers={"Authorization": f"Bearer {token}"}
                        )
                        require(
                            response.status_code == 200, "Monitoring failed during soak"
                        )
                        require(
                            response.json()["active_calls"] == 0,
                            "Monitoring retained active calls",
                        )
                        require(
                            "Soak question" not in response.text,
                            "Monitoring leaked research content",
                        )
                    if pdf and index % 10 == 0:
                        from sibyl.pdf_reader import read_pdf

                        result = await read_pdf(b"%PDF-invalid", 1000)
                        require(bool(result.get("error")), "Malformed PDF was accepted")
                        pdf_calls += 1

                for index in range(3):
                    await cycle(index)
                gc.collect()
                warm_fds = descriptor_count()
                warm_tasks = asyncio.all_tasks()
                tracemalloc.start()
                initial_memory = tracemalloc.get_traced_memory()[0]
                for index in range(rounds):
                    await cycle(index)
                    gc.collect()
                    files = [path for path in root.rglob("*") if path.is_file()]
                    sample = {
                        "round": index + 1,
                        "files": len(files),
                        "disk_bytes": sum(path.stat().st_size for path in files),
                        "open_descriptors": descriptor_count(),
                        "python_bytes": tracemalloc.get_traced_memory()[0],
                    }
                    measurements.append(sample)
                    require(
                        {path.relative_to(root).as_posix() for path in files}
                        == {
                            ".artifact.lock",
                            ".research/owner.lock",
                            ".research/jobs.sqlite3",
                        },
                        "Unexpected retained files",
                    )
                    require(
                        sample["disk_bytes"] < 1024 * 1024,
                        "Checkpoint disk usage grew beyond fixture bound",
                    )
                    require(
                        not (asyncio.all_tasks() - warm_tasks),
                        "Async tasks accumulated",
                    )
                    if warm_fds is not None:
                        require(
                            sample["open_descriptors"] <= warm_fds,
                            "File descriptors accumulated",
                        )
                memory_growth = measurements[-1]["python_bytes"] - initial_memory
                require(
                    memory_growth < 8 * 1024 * 1024,
                    "Retained Python allocations exceeded 8 MiB",
                )
                traced_peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                checkpoint = jobs.start("Backup recovery", [], "host", 1, "en", 3)
                expected = jobs.store.get(checkpoint["job_id"])
                await runtime.aclose()
                if monitor:
                    try:
                        await client.get(monitor_url)
                    except httpx.ConnectError:
                        pass
                    else:
                        raise RuntimeError("Monitor listener survived shutdown")
            restored_root = Path(directory) / "restored"
            (restored_root / ".research").mkdir(parents=True, mode=0o700)
            shutil.copy2(jobs.store.path, restored_root / ".research/jobs.sqlite3")
            restored = SibylRuntime(RuntimePolicy(output_root=restored_root), Config())
            try:
                require(
                    restored.research_jobs.store.get(checkpoint["job_id"]) == expected,
                    "Stopped-store backup changed checkpoint",
                )
                await restored.research_jobs.cancel(checkpoint["job_id"], delete=True)
            finally:
                await restored.aclose()
            await asyncio.sleep(0)
            gc.collect()
            require(
                not (asyncio.all_tasks() - baseline_tasks), "Tasks survived shutdown"
            )
            if baseline_children is not None:
                require(
                    child_processes() == baseline_children,
                    "Child processes survived shutdown",
                )
            final_fds = descriptor_count()
            if baseline_fds is not None:
                require(final_fds <= baseline_fds, "Descriptors survived shutdown")
            return {
                "passed": True,
                "rounds": rounds,
                "warmup_rounds": 3,
                "concurrent_workers": 2,
                "upstream": "injected read errors, cancellation, and timeouts; no public network/model calls",
                "monitor": monitor,
                "pdf_worker_calls": pdf_calls,
                "backup_restore": "passed with runtime stopped",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "retained_python_growth_bytes": memory_growth,
                "traced_peak_bytes": traced_peak,
                "final_open_descriptors": final_fds,
                "samples": measurements,
            }
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            await runtime.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="New JSON result file; refuses to overwrite"
    )
    args = parser.parse_args()
    if not 3 <= args.rounds <= 1000:
        parser.error("Use 3-1000 bounded rounds")
    result = asyncio.run(check(args.rounds, args.monitor, args.pdf))
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        result = {key: value for key, value in result.items() if key != "samples"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
