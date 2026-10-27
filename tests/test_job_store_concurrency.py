"""Shared research stores across real local processes. POSIX only: Windows
locking paths are not exercised here and need a real Windows run."""

import asyncio
import errno
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from research_fixtures import evidence_bundle

from sibyl.async_cache import CapacityError
from sibyl.config import Config
from sibyl.job_store import HistoryFull, JobStore, OwnerUnknown, StoreBusy
from sibyl.mcp_server import create_server
from sibyl.research_errors import ResearchRequestError
from sibyl.research_jobs import CheckpointDiscarded, ResultPending
from sibyl.runtime import RuntimePolicy, SibylRuntime

ROOT = Path(__file__).resolve().parents[1]

# Claim a job, report it, then wait to be killed without closing the store.
OWNER = """
import sys, time
from pathlib import Path
from sibyl.job_store import JobStore
store = JobStore(Path(sys.argv[1]))
job = store.get(sys.argv[2])
store.claim(job)
print(store.instance_id, job['revision'], flush=True)
time.sleep(120)
"""

# Start at a shared instant so the processes race on the same store.
RACER = """
import sys, time
from pathlib import Path
from sibyl.async_cache import CapacityError
from sibyl.job_store import JobStore
time.sleep(max(0.0, float(sys.argv[2]) - time.time()))
store = JobStore(Path(sys.argv[1]))
if sys.argv[3] == 'create':
    try:
        store.create({'question': 'q', 'mode': 'host', 'depth': 1, 'language': 'en',
                      'status': 'awaiting_input', 'notebook': {}, 'report': None,
                      'events': [], 'elapsed_seconds': 0.0})
        print('created')
    except CapacityError:
        print('full')
store.close()
"""


def payload(status="awaiting_input"):
    return {
        "question": "q",
        "mode": "host",
        "depth": 1,
        "language": "en",
        "status": status,
        "notebook": {"attempts": [{"status": "running"}]},
        "report": None,
        "events": [],
        "elapsed_seconds": 0.0,
    }


@unittest.skipUnless(os.name == "posix", "POSIX flock semantics")
class TestSharedJobStore(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.stores = []
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.communicate()
        for store in self.stores:
            store.close()
        self.directory.cleanup()

    def store(self):
        store = JobStore(self.root)
        self.stores.append(store)
        return store

    def spawn(self, script, *args):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(self.root), *map(str, args)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(process)
        return process

    def orphan(self):
        """A job claimed by a process that was then killed without cleanup."""
        job = self.store().create(payload())
        owner = self.spawn(OWNER, job["job_id"])
        instance_id, revision = owner.stdout.readline().split()
        owner.kill()
        owner.wait(5)
        return job["job_id"], instance_id, int(revision)

    def race(self, count, mode="open"):
        start = time.time() + 1.0
        racers = [self.spawn(RACER, start, mode) for _ in range(count)]
        outputs = []
        for racer in racers:
            out, err = racer.communicate(timeout=30)
            self.assertEqual(racer.returncode, 0, err)
            outputs.append(out.strip())
        return outputs

    def test_concurrent_takeover_of_one_orphan_recovers_once(self):
        job_id, instance_id, revision = self.orphan()
        lock = self.root / ".research" / "instances" / f"{instance_id}.lock"
        self.assertTrue(lock.exists(), "a killed owner leaves its lock file")
        self.race(4)
        job = self.store().get(job_id)
        self.assertEqual(job["status"], "interrupted")
        self.assertEqual(job["revision"], revision + 1)
        self.assertNotIn("owner", job)
        self.assertEqual(job["notebook"]["attempts"][0]["status"], "failed")
        self.assertEqual(
            sum("Runtime stopped" in event["message"] for event in job["events"]), 1
        )
        self.assertFalse(lock.exists(), "stale instance lock was not removed")

    def test_concurrent_creation_cannot_exceed_capacity(self):
        store = self.store()
        for _ in range(JobStore.MAX_JOBS - 1):
            store.create(payload())
        outputs = self.race(4, "create")
        self.assertEqual(sorted(outputs), ["created", "full", "full", "full"])
        self.assertEqual(len(store.list_jobs()), JobStore.MAX_JOBS)

    def test_live_job_is_read_only_elsewhere_and_recovered_after_exit(self):
        owner, observer = self.store(), self.store()
        job = owner.create(payload())
        owner.claim(job)
        seen = observer.get(job["job_id"])
        self.assertTrue(observer.running_elsewhere(seen))
        self.assertIsNone(observer.recover(seen))
        with self.assertRaisesRegex(ValueError, "Cancel a running job"):
            observer.delete(job["job_id"])
        self.assertEqual(observer.get(job["job_id"])["revision"], job["revision"])
        owner.close()
        self.assertFalse(observer.running_elsewhere(seen))
        self.assertEqual(observer.recover(seen)["status"], "interrupted")

    def test_worker_capacity_is_shared_by_every_runtime(self):
        one, two, three = self.store(), self.store(), self.store()
        jobs = [one.create(payload()) for _ in range(3)]
        one.claim(jobs[0])
        two.claim(jobs[1])
        with self.assertRaisesRegex(CapacityError, "across sessions"):
            three.claim(jobs[2])
        one.close()
        three.claim(jobs[2])

    def test_lock_check_errors_never_mean_the_owner_died(self):
        owner, observer = self.store(), self.store()
        job = owner.create(payload())
        owner.claim(job)
        seen = observer.get(job["job_id"])
        real_open = os.open
        lock_name = f"{owner.instance_id}.lock"

        def denied(path, *args, **kwargs):
            if str(path).endswith(lock_name):
                raise PermissionError(errno.EACCES, "denied")
            return real_open(path, *args, **kwargs)

        with mock.patch("sibyl.job_store.os.open", denied):
            with self.assertRaises(OwnerUnknown):
                observer.recover(seen)
            self.assertTrue(observer.running_elsewhere(seen))
        with mock.patch("fcntl.flock", side_effect=OSError(errno.EIO, "io")):
            with self.assertRaises(OwnerUnknown):
                observer.recover(seen)
        self.assertEqual(observer.get(job["job_id"])["status"], "running")

    def test_corrupt_owner_record_is_not_taken_over(self):
        store = self.store()
        job = store.create(payload())
        job.update(status="running", owner={"instance_id": "../x", "claimed_at": 0})
        store.save(job)
        with self.assertRaises(OwnerUnknown):
            self.store().recover(store.get(job["job_id"]))

    def test_stale_lock_cleanup_keeps_live_and_fresh_pending_files(self):
        live = self.store()
        instances = live.instances
        stale = instances / ("a" * 32 + ".lock")
        stale.write_text("1\n")
        fresh = instances / (".pending-" + "b" * 32)
        fresh.write_text("1\n")
        old = instances / (".pending-" + "c" * 32)
        old.write_text("1\n")
        os.utime(old, (time.time() - 3600, time.time() - 3600))
        self.store()
        self.assertFalse(stale.exists())
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue((instances / f"{live.instance_id}.lock").exists())

    def test_older_exclusive_runtime_blocks_startup_and_vice_versa(self):
        import fcntl

        research = self.root / ".research"
        research.mkdir(mode=0o700)
        legacy = os.open(research / "owner.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(CapacityError, "older Sibyl runtime"):
                JobStore(self.root)
            fcntl.flock(legacy, fcntl.LOCK_UN)
            self.store()
            with self.assertRaises(BlockingIOError):
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(legacy)

    def test_startup_recovery_tolerates_races_and_busy_store(self):
        self.store().create(payload("running"))
        for failure in (
            ValueError("Research job not found."),
            StoreBusy("busy"),
            OwnerUnknown(),
        ):
            with mock.patch.object(JobStore, "recover", side_effect=failure):
                self.store()
        with mock.patch.object(JobStore, "list_jobs", side_effect=StoreBusy("busy")):
            self.store()

    def test_full_database_error_is_not_masked_by_rollback(self):
        store = self.store()
        with mock.patch.object(JobStore, "MAX_PAGES", 12):
            with self.assertRaisesRegex(sqlite3.OperationalError, "full"):
                for _ in range(JobStore.MAX_JOBS):
                    store.create({**payload(), "question": "x" * 4000})

    def test_claim_after_concurrent_delete_reports_missing_job(self):
        one, two = self.store(), self.store()
        job = one.create(payload())
        two.delete(job["job_id"])
        with self.assertRaisesRegex(ValueError, "not found"):
            one.claim(job)


@unittest.skipUnless(os.name == "posix", "POSIX flock semantics")
class TestCrossSessionJobs(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        policy = RuntimePolicy(output_root=Path(self.directory.name))
        self.owner = SibylRuntime(policy, Config())
        self.other = SibylRuntime(policy, Config())

    async def asyncTearDown(self):
        await self.owner.aclose()
        await self.other.aclose()
        self.directory.cleanup()

    async def test_other_session_reads_but_cannot_change_a_running_job(self):
        jobs = self.owner.research_jobs
        public = jobs.start("Question", [], "host", 1, "en", 3)
        job = jobs.store.get(public["job_id"])
        jobs.store.claim(job)
        other = self.other.research_jobs
        seen = other.get(job["job_id"])
        self.assertEqual(seen["status"], "running")
        self.assertTrue(seen["running_elsewhere"])
        self.assertNotIn("owner", seen)
        for change in (
            lambda: other.assess(
                job["job_id"], job["revision"], "Q0", "unresolved", [], [], "c", "n"
            ),
            lambda: other.complete(
                job["job_id"], job["revision"], "", [], "", [], True
            ),
        ):
            with self.assertRaises(ResearchRequestError) as raised:
                change()
            self.assertEqual(raised.exception.code, "job_running_elsewhere")
        for delete in (False, True):
            with self.assertRaises(ResearchRequestError) as raised:
                await other.cancel(job["job_id"], delete)
            self.assertEqual(raised.exception.code, "job_running_elsewhere")
        with self.assertRaises(ResearchRequestError) as raised:
            await other.continue_job(job["job_id"], "query")
        self.assertEqual(raised.exception.code, "job_running_elsewhere")
        self.assertEqual(other.store.get(job["job_id"])["revision"], job["revision"])
        server = create_server(self.other, agent_mode=True)
        response = await server.call_tool(
            "assess_research",
            {
                "job_id": job["job_id"],
                "revision": job["revision"],
                "question_id": "Q0",
                "status": "unresolved",
                "conclusion": "c",
                "counterevidence_note": "n",
            },
        )
        self.assertTrue(response.isError)
        error = response.structuredContent
        self.assertEqual(error["error"]["code"], "job_running_elsewhere")
        self.assertEqual(error["recovery"]["action"], "wait_for_other_session")
        self.assertEqual(error["job_status"], "running")
        await self.owner.aclose()
        resumed = await other.cancel(job["job_id"])
        self.assertEqual(resumed["status"], "cancelled")
        self.assertFalse(resumed["running_elsewhere"])

    def block_store(self, jobs):
        """Hold the write lock from outside, as another session would."""
        blocker = sqlite3.connect(jobs.store.path, isolation_level=None)
        self.addCleanup(blocker.close)
        return blocker

    def counting_retriever(self, blocker=None):
        calls = []

        async def retrieve(*args, **kwargs):
            calls.append(args)
            if blocker is not None and not blocker.in_transaction:
                blocker.execute("BEGIN IMMEDIATE")
            return evidence_bundle("busy", 2)

        return calls, retrieve

    async def test_pending_result_guidance_never_repeats_the_retrieval(self):
        jobs = self.owner.research_jobs
        server = create_server(self.owner, agent_mode=True)
        job = jobs.start("Question", [], "host", 1, "en", 3)
        blocker = self.block_store(jobs)
        calls, retrieve = self.counting_retriever(blocker)
        arguments = {"job_id": job["job_id"], "query": "query", "question_id": "Q0"}
        with (
            mock.patch.object(JobStore, "BUSY_TIMEOUT", 0.1),
            mock.patch("sibyl.research_jobs.gather_source_bundle", retrieve),
        ):
            response = await server.call_tool("continue_research", arguments)
            self.assertTrue(response.isError)
            error = response.structuredContent
            self.assertEqual(error["error"]["code"], "result_pending")
            self.assertEqual(error["recovery"]["action"], "read_research")
            self.assertIn("Do not", error["recovery"]["instructions"])
            read = error["recovery"]["read"]
            self.assertEqual(read["tool"], "get_research")
            # Following the guidance while the store is still busy shows the
            # held result and says it is not persisted.
            held = await server.call_tool(read["tool"], read["arguments"])
            self.assertFalse(held.isError)
            self.assertTrue(held.structuredContent["unsaved_result"])
            self.assertEqual(held.structuredContent["latest_attempt"]["status"], "ok")
            self.assertTrue(
                self.other.research_jobs.get(job["job_id"])["running_elsewhere"]
            )
            # A blind retry is refused instead of searching again.
            retried = await server.call_tool("continue_research", arguments)
            self.assertEqual(
                retried.structuredContent["error"]["code"], "result_pending"
            )
            blocker.execute("COMMIT")
            saved = await server.call_tool(read["tool"], read["arguments"])
        self.assertEqual(len(calls), 1)
        brief = saved.structuredContent
        # The held view and the error envelope keep the stored revision; saving
        # moves past it, so a request against the held view must reload first.
        self.assertEqual(error["current_revision"], held.structuredContent["revision"])
        self.assertGreater(brief["revision"], held.structuredContent["revision"])
        stale = await server.call_tool(
            "assess_research",
            {
                "job_id": job["job_id"],
                "revision": held.structuredContent["revision"],
                "question_id": "Q0",
                "status": "unresolved",
                "conclusion": "c",
                "counterevidence_note": "n",
            },
        )
        self.assertEqual(stale.structuredContent["error"]["code"], "stale_revision")
        self.assertFalse(brief["unsaved_result"])
        self.assertEqual(brief["latest_attempt"]["status"], "ok")
        stored = jobs.store.get(job["job_id"])
        self.assertEqual(stored["status"], "awaiting_input")
        self.assertNotIn("owner", stored)
        self.assertEqual(len(stored["notebook"]["bundles"]), 1)
        self.assertFalse(
            any("size limit" in event["message"] for event in stored["events"])
        )
        self.assertFalse(
            self.other.research_jobs.get(job["job_id"])["running_elsewhere"]
        )

    async def test_busy_before_retrieval_is_safe_to_retry(self):
        jobs = self.owner.research_jobs
        server = create_server(self.owner, agent_mode=True)
        job = jobs.start("Question", [], "host", 1, "en", 3)
        calls, retrieve = self.counting_retriever()
        arguments = {"job_id": job["job_id"], "query": "query", "question_id": "Q0"}
        with mock.patch("sibyl.research_jobs.gather_source_bundle", retrieve):
            with mock.patch.object(JobStore, "claim", side_effect=StoreBusy("busy")):
                response = await server.call_tool("continue_research", arguments)
            self.assertEqual(response.structuredContent["error"]["code"], "store_busy")
            self.assertEqual(
                response.structuredContent["recovery"]["action"], "retry_later"
            )
            self.assertEqual(calls, [])
            retried = await server.call_tool("continue_research", arguments)
        self.assertFalse(retried.isError)
        self.assertEqual(len(calls), 1)

    async def test_shutdown_reports_a_result_it_could_not_save(self):
        jobs = self.owner.research_jobs
        job = jobs.start("Question", [], "host", 1, "en", 3)
        blocker = self.block_store(jobs)
        _, retrieve = self.counting_retriever(blocker)
        with (
            mock.patch.object(JobStore, "BUSY_TIMEOUT", 0.1),
            mock.patch("sibyl.research_jobs.gather_source_bundle", retrieve),
        ):
            with self.assertRaises(ResultPending):
                await jobs.continue_job(job["job_id"], "query", question_id="Q0")
            with self.assertLogs("sibyl.research_jobs", "WARNING") as logs:
                await self.owner.aclose()
        blocker.execute("COMMIT")
        message = " ".join(logs.output)
        self.assertIn("NOT saved", message)
        self.assertIn("lost when this process exits", message)
        self.assertIn(job["job_id"], message)
        recovered = self.other.research_jobs.get(job["job_id"])
        self.assertEqual(recovered["status"], "interrupted")
        self.assertFalse(recovered["running_elsewhere"])
        self.assertFalse(recovered["evidence"])

    async def test_cancel_does_not_overwrite_a_claim_made_while_it_waited(self):
        jobs, other = self.owner.research_jobs, self.other.research_jobs
        job = jobs.start("Question", [], "host", 1, "en", 3)
        started = asyncio.Event()

        async def blocked(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        checkpoint = jobs._checkpoint

        def checkpoint_then_claim_elsewhere(record):
            checkpoint(record)
            # Another session claims the job right after this cancellation saves.
            other.store.claim(other.store.get(record["job_id"]))

        with (
            mock.patch("sibyl.research_jobs.gather_source_bundle", blocked),
            mock.patch.object(jobs, "_checkpoint", checkpoint_then_claim_elsewhere),
        ):
            task = asyncio.create_task(
                jobs.continue_job(job["job_id"], "query", question_id="Q0")
            )
            await asyncio.wait_for(started.wait(), 5)
            for delete in (False, True):
                with self.subTest(delete=delete):
                    with self.assertRaises(ResearchRequestError) as raised:
                        await jobs.cancel(job["job_id"], delete)
                    self.assertEqual(raised.exception.code, "job_running_elsewhere")
            await asyncio.gather(task, return_exceptions=True)
        stored = jobs.store.get(job["job_id"])
        self.assertEqual(stored["status"], "running")
        self.assertEqual(stored["owner"]["instance_id"], other.store.instance_id)

    async def test_busy_creation_leaves_no_job_to_duplicate(self):
        jobs = self.owner.research_jobs
        blocker = self.block_store(jobs)
        blocker.execute("BEGIN IMMEDIATE")
        with mock.patch.object(JobStore, "BUSY_TIMEOUT", 0.1):
            with self.assertRaises(StoreBusy):
                jobs.start("Question", [], "host", 1, "en", 3)
        blocker.execute("COMMIT")
        self.assertEqual(jobs.store.list_jobs(), [])
        transaction = JobStore.transaction
        with mock.patch.object(
            JobStore, "transaction", autospec=True, side_effect=transaction
        ) as writes:
            jobs.start("Question", [], "host", 1, "en", 3)
        self.assertEqual(writes.call_count, 1)

    async def test_failed_result_save_releases_the_claim(self):
        jobs, other = self.owner.research_jobs, self.other.research_jobs
        job = jobs.start("Question", [], "host", 1, "en", 3)
        _, retrieve = self.counting_retriever()
        full = sqlite3.OperationalError("database or disk is full")
        with (
            mock.patch("sibyl.research_jobs.gather_source_bundle", retrieve),
            mock.patch.object(jobs.store, "save", side_effect=full),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "full"):
                await jobs.continue_job(job["job_id"], "query", question_id="Q0")
        # Releasing needed a write too, so it waits for this session's next call.
        self.assertTrue(other.get(job["job_id"])["running_elsewhere"])
        jobs.list_jobs()
        released = other.get(job["job_id"])
        self.assertFalse(released["running_elsewhere"])
        self.assertEqual(released["status"], "interrupted")
        spare = [other.store.create(payload()) for _ in range(2)]
        for record in spare:
            other.store.claim(record)

    async def test_held_result_is_saved_by_a_call_on_another_job(self):
        jobs = self.owner.research_jobs
        held = jobs.start("Held", [], "host", 1, "en", 3)
        unrelated = jobs.start("Other", [], "host", 1, "en", 3)
        blocker = self.block_store(jobs)
        _, retrieve = self.counting_retriever(blocker)
        with (
            mock.patch.object(JobStore, "BUSY_TIMEOUT", 0.1),
            mock.patch("sibyl.research_jobs.gather_source_bundle", retrieve),
        ):
            with self.assertRaises(ResultPending):
                await jobs.continue_job(held["job_id"], "query", question_id="Q0")
        blocker.execute("COMMIT")
        jobs.get(unrelated["job_id"])
        stored = jobs.store.get(held["job_id"])
        self.assertEqual(stored["status"], "awaiting_input")
        self.assertEqual(len(stored["notebook"]["bundles"]), 1)

    async def test_held_revision_cannot_match_a_release_without_the_result(self):
        jobs = self.owner.research_jobs
        job = jobs.start("Question", [], "host", 1, "en", 3)
        blocker = self.block_store(jobs)
        _, retrieve = self.counting_retriever(blocker)
        with (
            mock.patch.object(JobStore, "BUSY_TIMEOUT", 0.1),
            mock.patch("sibyl.research_jobs.gather_source_bundle", retrieve),
        ):
            with self.assertRaises(ResultPending):
                await jobs.continue_job(job["job_id"], "query", question_id="Q0")
            held = jobs.get(job["job_id"])
        blocker.execute("COMMIT")
        self.assertTrue(held["unsaved_result"])
        # Saving fails for good, so the claim is released without the result.
        with mock.patch.object(jobs, "_checkpoint", side_effect=RuntimeError("io")):
            with self.assertLogs("sibyl.research_jobs", "WARNING"):
                jobs.list_jobs()
        released = jobs.get(job["job_id"])
        self.assertEqual(released["status"], "interrupted")
        self.assertFalse(released["evidence"])
        with self.assertRaises(ResearchRequestError) as raised:
            jobs.assess(
                job["job_id"], held["revision"], "Q0", "unresolved", [], [], "c", "n"
            )
        self.assertEqual(raised.exception.code, "stale_revision")

    async def test_oversized_held_result_is_logged_as_saved_without_evidence(self):
        jobs = self.owner.research_jobs
        job = jobs.start("Question", [], "host", 1, "en", 3)
        jobs._pending[job["job_id"]] = jobs.store.get(job["job_id"])
        with mock.patch.object(
            jobs, "_checkpoint", side_effect=CheckpointDiscarded("limit")
        ):
            with self.assertLogs("sibyl.research_jobs", "WARNING") as logs:
                jobs.list_jobs()
        message = " ".join(logs.output)
        self.assertIn("prior state was saved", message)
        self.assertNotIn("NOT saved", message)
        self.assertFalse(jobs._stale_claims)

    async def test_busy_cancel_says_the_local_retrieval_already_stopped(self):
        jobs = self.owner.research_jobs
        job = jobs.start("Question", [], "host", 1, "en", 3)
        started = asyncio.Event()

        async def blocked(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        save = jobs.store.save

        def busy_when_cancelling(record, **kwargs):
            if record["status"] == "cancelled":
                raise StoreBusy("busy")
            return save(record, **kwargs)

        with mock.patch("sibyl.research_jobs.gather_source_bundle", blocked):
            task = asyncio.create_task(
                jobs.continue_job(job["job_id"], "query", question_id="Q0")
            )
            await asyncio.wait_for(started.wait(), 5)
            with mock.patch.object(
                jobs.store, "save", side_effect=busy_when_cancelling
            ):
                with self.assertRaisesRegex(StoreBusy, "stopped") as raised:
                    await jobs.cancel(job["job_id"])
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(raised.exception.code, "store_busy")
        self.assertEqual(jobs.store.get(job["job_id"])["status"], "interrupted")
        self.assertEqual((await jobs.cancel(job["job_id"]))["status"], "cancelled")

    async def test_busy_delete_says_the_local_retrieval_already_stopped(self):
        jobs = self.owner.research_jobs
        job = jobs.start("Question", [], "host", 1, "en", 3)
        started = asyncio.Event()

        async def blocked(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        with mock.patch("sibyl.research_jobs.gather_source_bundle", blocked):
            task = asyncio.create_task(
                jobs.continue_job(job["job_id"], "query", question_id="Q0")
            )
            await asyncio.wait_for(started.wait(), 5)
            with mock.patch.object(jobs.store, "delete", side_effect=StoreBusy("busy")):
                with self.assertRaisesRegex(StoreBusy, "stopped.*deleting"):
                    await jobs.cancel(job["job_id"], delete=True)
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(
            (await jobs.cancel(job["job_id"], delete=True))["status"], "deleted"
        )

    async def test_store_busy_guidance_does_not_claim_nothing_changed(self):
        server = create_server(self.other, agent_mode=True)
        with mock.patch.object(JobStore, "create", side_effect=StoreBusy("busy")):
            response = await server.call_tool("start_research", {"question": "Q"})
        instructions = response.structuredContent["recovery"]["instructions"]
        self.assertNotIn("nothing was changed", instructions)
        self.assertIn("whether part of it already took effect", instructions)

    async def test_transient_capacity_errors_carry_retry_guidance(self):
        server = create_server(self.other, agent_mode=True)
        for error, code, action in (
            (StoreBusy("busy"), "store_busy", "retry_later"),
            (HistoryFull("full"), "history_full", "delete_finished_jobs"),
        ):
            with mock.patch.object(JobStore, "create", side_effect=error):
                response = await server.call_tool(
                    "start_research", {"question": "Question"}
                )
            self.assertTrue(response.isError)
            self.assertEqual(response.structuredContent["error"]["code"], code)
            self.assertEqual(response.structuredContent["recovery"]["action"], action)


if __name__ == "__main__":
    unittest.main()
