"""Bounded PDF subprocess lifecycle without importing a parser into the host."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import weakref
from pathlib import Path

_gates = weakref.WeakKeyDictionary()
_WORKER = str(Path(__file__).with_name("pdf_worker.py"))


async def _watch_memory(process):
    # macOS does not enforce RLIMIT_AS; sampled RSS is a kill threshold, not a hard allocation cap.
    while process.returncode is None:
        probe = await asyncio.create_subprocess_exec(
            "/bin/ps",
            "-o",
            "rss=",
            "-p",
            str(process.pid),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={},
        )
        try:
            output, _ = await probe.communicate()
        finally:
            if probe.returncode is None:
                probe.kill()
            await probe.wait()
        if process.returncode is not None:
            return
        if not output.strip().isdigit() or int(output) > 512 * 1024:
            process.kill()
            return
        await asyncio.sleep(0.1)


async def read_pdf(data: bytes, max_chars: int) -> dict:
    if os.name != "posix":
        return {"error": "Resource-limited PDF extraction requires a POSIX host."}
    if len(data) > 2 * 1024 * 1024 or not data.startswith(b"%PDF-"):
        return {"error": "Invalid or oversized PDF."}
    loop = asyncio.get_running_loop()
    gate = _gates.setdefault(loop, asyncio.Semaphore(2))
    async with gate:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            _WORKER,
            str(max(200, min(max_chars, 60000))),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={},
        )
        watcher = (
            asyncio.create_task(_watch_memory(process))
            if sys.platform == "darwin"
            else None
        )
        if watcher is not None:

            def checked(task):
                if (
                    not task.cancelled()
                    and task.exception() is not None
                    and process.returncode is None
                ):
                    process.kill()

            watcher.add_done_callback(checked)
        try:
            output, _ = await asyncio.wait_for(process.communicate(data), timeout=6)
            if process.returncode != 0 or len(output) > 400000:
                return {"error": "PDF worker failed or exceeded a resource limit."}
            return json.loads(output)
        except asyncio.TimeoutError:
            return {"error": "PDF extraction timed out."}
        except (ValueError, BrokenPipeError):
            return {"error": "PDF extraction failed."}
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            if process.returncode is None:
                process.kill()
            await process.wait()
