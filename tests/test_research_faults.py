"""Crash, failed persistence, and bounded concurrency recovery without network calls."""

import asyncio
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_researcher import evidence_bundle

from sibyl.async_cache import CapacityError
from sibyl.config import Config
from sibyl.job_store import JobStore
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.storage import ArtifactStore

ROOT = Path(__file__).resolve().parents[1]
CRASH_WORKER = """
import sys, time
from pathlib import Path
from sibyl.config import Config
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.research_state import ResearchNotebook
runtime = SibylRuntime(RuntimePolicy(output_root=Path(sys.argv[1])), Config())
jobs = runtime.research_jobs
public = jobs.start('Crash recovery', [], 'host', 1, 'en', 3)
job = jobs.store.get(public['job_id'])
notebook = ResearchNotebook.from_dict(job['notebook'])
notebook.begin_search('reserved before dispatch')
notebook.reserve_model_call(100)
job.update(status='running', notebook=notebook.to_dict())
jobs.store.save(job)
print(job['job_id'], flush=True)
time.sleep(60)
"""


class TestResearchFaults(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.runtime = SibylRuntime(RuntimePolicy(output_root=self.root), Config())
        self.jobs = self.runtime.research_jobs

    async def asyncTearDown(self):
        await self.runtime.aclose()
        self.directory.cleanup()

    def start(self):
        return self.jobs.start("Question", [], "host", 1, "en", 3)

    async def test_write_failure_before_dispatch_never_calls_retriever(self):
        job = self.start()
        with (
            mock.patch.object(
                self.jobs.store,
                "save",
                side_effect=sqlite3.OperationalError("disk is full"),
            ),
            mock.patch(
                "sibyl.research_jobs.gather_source_bundle", mock.AsyncMock()
            ) as fetch,
        ):
            with self.assertRaises(sqlite3.OperationalError):
                await self.jobs.continue_job(job["job_id"], "focused query")
        fetch.assert_not_called()
        restored = self.jobs.get(job["job_id"])
        self.assertEqual(restored["revision"], job["revision"])
        self.assertEqual(restored["remaining_steps"], 3)
        self.assertFalse(self.jobs.tasks)

    async def test_write_failure_after_dispatch_retains_reservation_on_restart(self):
        job = self.start()
        save = self.jobs.store.save
        calls = 0

        def fail_result(record, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise sqlite3.OperationalError("disk is full")
            save(record, **kwargs)

        with (
            mock.patch.object(self.jobs.store, "save", side_effect=fail_result),
            mock.patch(
                "sibyl.research_jobs.gather_source_bundle",
                mock.AsyncMock(return_value=evidence_bundle()),
            ),
        ):
            with self.assertRaises(sqlite3.OperationalError):
                await self.jobs.continue_job(job["job_id"], "focused query")
        self.assertFalse(self.jobs.tasks)
        await self.runtime.aclose()
        self.runtime = SibylRuntime(RuntimePolicy(output_root=self.root), Config())
        self.jobs = self.runtime.research_jobs
        restored = self.jobs.get(job["job_id"])
        self.assertEqual(restored["status"], "interrupted")
        self.assertEqual(restored["remaining_steps"], 2)
        self.assertEqual(restored["attempts"][0]["status"], "failed")
        self.assertFalse(restored["evidence"])

    async def test_transaction_abort_does_not_advance_revision(self):
        record = self.jobs.store.get(self.start()["job_id"])
        original = self.jobs.store.get(record["job_id"])
        with self.jobs.store.connection() as db:
            db.execute(
                "CREATE TRIGGER reject_write BEFORE UPDATE ON jobs BEGIN SELECT RAISE(ABORT, 'write rejected'); END"
            )
        record["status"] = "cancelled"
        with self.assertRaises(sqlite3.IntegrityError):
            self.jobs.store.save(record)
        self.assertEqual(self.jobs.store.get(record["job_id"]), original)
        self.assertEqual(record["revision"], original["revision"])
        with self.jobs.store.connection() as db:
            db.execute("DROP TRIGGER reject_write")
        self.jobs.store.save(record)
        self.assertEqual(record["revision"], original["revision"] + 1)

    async def test_repeated_capacity_and_cancellation_leave_no_workers(self):
        for _ in range(8):
            started = asyncio.Event()
            count = 0

            async def blocked(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    started.set()
                await asyncio.Event().wait()

            jobs = [self.start() for _ in range(3)]
            with mock.patch("sibyl.research_jobs.gather_source_bundle", blocked):
                tasks = [
                    asyncio.create_task(self.jobs.continue_job(j["job_id"], "query"))
                    for j in jobs[:2]
                ]
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    with self.assertRaises(CapacityError):
                        await self.jobs.continue_job(jobs[2]["job_id"], "query")
                    for job in jobs[:2]:
                        await self.jobs.cancel(job["job_id"])
                    self.assertTrue(all(task.cancelled() for task in tasks))
                    self.assertFalse(self.jobs.tasks)
                    self.assertEqual(
                        self.jobs.get(jobs[2]["job_id"])["remaining_steps"], 3
                    )
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    @unittest.skipUnless(os.name == "posix", "POSIX kill/owner-lock recovery")
    async def test_process_kill_releases_owner_and_preserves_consumed_budgets(self):
        await self.runtime.aclose()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            CRASH_WORKER,
            str(self.root),
            cwd=ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            line = await asyncio.wait_for(process.stdout.readline(), 10)
            self.assertTrue(
                line, "Crash worker failed before committing its reservation"
            )
            job_id = line.decode().strip()
            with self.assertRaises(CapacityError):
                JobStore(self.root)
            process.kill()
            await asyncio.wait_for(process.wait(), 5)
        finally:
            if process.returncode is None:
                process.kill()
            await process.communicate()
        self.runtime = SibylRuntime(RuntimePolicy(output_root=self.root), Config())
        restored = self.runtime.research_jobs.get(job_id)
        self.assertEqual(restored["status"], "interrupted")
        self.assertEqual(restored["remaining_steps"], 2)
        self.assertEqual(restored["model_calls"], 1)
        self.assertEqual(restored["reserved_output_tokens"], 100)
        self.assertEqual(restored["attempts"][0]["status"], "failed")
        self.assertFalse(self.runtime.research_jobs.tasks)

    async def test_failed_artifact_publication_cleans_staging_file(self):
        store = ArtifactStore(self.root)
        with mock.patch(
            "sibyl.storage.os.replace", side_effect=OSError("disk failure")
        ):
            with self.assertRaises(OSError):
                store.write(b"report", "report", ".md")
        self.assertFalse(list(self.root.glob(".sibyl-*")))
        self.assertFalse(list(self.root.glob("sibyl_*")))
