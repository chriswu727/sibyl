"""Shared research stores across real local processes. POSIX only: Windows
locking paths are not exercised here and need a real Windows run."""

import errno
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from sibyl.async_cache import CapacityError
from sibyl.config import Config
from sibyl.job_store import JobStore, OwnerUnknown
from sibyl.mcp_server import create_server
from sibyl.research_errors import ResearchRequestError
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


if __name__ == "__main__":
    unittest.main()
