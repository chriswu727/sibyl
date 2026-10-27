"""Bounded local SQLite checkpoints shared by concurrent local runtimes.

Each runtime holds a private instance lock for its whole lifetime. A running job
records the instance that claimed it; other runtimes may read that job but not
change it, and may recover it only once the owner's lock is free. Claims,
recovery, creation and deletion are conditional writes inside immediate
transactions, so when two processes race exactly one of them succeeds.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import sqlite3
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import uuid4

from .async_cache import CapacityError
from .job_models import JobOwner, JobPayload, JobRecord
from .research_errors import ResearchRequestError
from .storage import ArtifactStore

logger = logging.getLogger(__name__)

_ID = re.compile(r"[a-f0-9]{32}")
# errno values meaning "another open file description holds the lock".
_HELD = (
    {errno.EACCES, errno.EDEADLK}
    if sys.platform == "win32"
    else {errno.EAGAIN, errno.EWOULDBLOCK}
)


class StoreBusy(CapacityError):
    """Another session held the database too long; nothing was written. Retry."""

    code = "store_busy"


class CheckpointTooLarge(CapacityError):
    code = "capacity_exceeded"


class HistoryFull(CapacityError):
    code = "history_full"


class WorkerCapacity(CapacityError):
    code = "worker_capacity"


class RevisionConflict(ResearchRequestError):
    def __init__(self) -> None:
        super().__init__(
            "Research revision changed; reload the job.", code="stale_revision"
        )


class OwnerUnknown(ResearchRequestError):
    """The owner lock of a running job could not be checked; never take it over."""

    def __init__(self) -> None:
        super().__init__(
            "Could not determine whether the session running this job is alive; "
            "the job was not taken over. Check the research directory permissions.",
            code="owner_unknown",
        )


def _try_lock(descriptor: int) -> bool:
    """Take an exclusive non-blocking lock; False only when another holder has it."""
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _HELD:
            return False
        raise
    return True


class JobStore:
    MAX_JOBS = 64
    MAX_PAYLOAD_BYTES = 1024 * 1024
    RECOVERY_RESERVE_BYTES = 64 * 1024
    MAX_PAGES = 16384
    # Active retrievals across every runtime sharing this store, not per process.
    MAX_RUNNING = 2
    # Store calls block the event loop while waiting, so keep this short.
    BUSY_TIMEOUT = 2.0
    STALE_PENDING_SECONDS = 60.0

    def __init__(self, root: Path) -> None:
        self.directory = ArtifactStore(root, ".research").root
        self.instances = self.directory / "instances"
        self.instance_id = ""
        self._legacy_lock: int | None = None
        self._instance_lock: int | None = None
        self._instance_path: Path | None = None
        try:
            self._legacy_lock = self._exclude_legacy_owner()
            self._register_instance()
            self.path = self.directory / "jobs.sqlite3"
            os.close(self._private_file(self.path))
            with self.connection() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA auto_vacuum=FULL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS jobs "
                    "(id TEXT PRIMARY KEY, payload TEXT NOT NULL, revision INTEGER NOT NULL, updated REAL NOT NULL)"
                )
            self._recover_orphans()
            self._remove_stale_instance_locks()
        except BaseException:
            self.close()
            raise

    def _recover_orphans(self) -> None:
        """Best-effort start-up recovery; a job left running is retried on access."""
        try:
            jobs = self.list_jobs()
        except StoreBusy:
            logger.warning("Skipped start-up recovery: research store busy")
            return
        for job in jobs:
            if job["status"] != "running":
                continue
            try:
                self.recover(job)
            except (CapacityError, ValueError) as exc:
                # Owner unknown, store busy, or deleted by another session.
                logger.warning("Skipped recovery of %s: %s", job["job_id"], exc)

    @staticmethod
    def _private_file(path: Path, flags: int = os.O_CREAT) -> int:
        if path.is_symlink():
            raise ValueError("Research state files must not be symlinks.")
        descriptor = os.open(
            path, os.O_RDWR | flags | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        if hasattr(os, "fchmod"):  # Windows has it only from Python 3.13.
            os.fchmod(descriptor, 0o600)
        return descriptor

    def _exclude_legacy_owner(self) -> int | None:
        """Refuse to share a directory with an older runtime that owns it exclusively.

        Older versions take an exclusive lock on owner.lock. Current runtimes hold a
        shared lock on POSIX, so neither generation can start beside the other.
        Windows has no shared lock: the check is made once at start-up.
        """
        descriptor = self._private_file(self.directory / "owner.lock")
        message = "An older Sibyl runtime owns this research directory; stop it first."
        try:
            if sys.platform == "win32":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"0")
                if not _try_lock(descriptor):
                    raise CapacityError(message)
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                os.close(descriptor)
                return None
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in _HELD:
                    raise CapacityError(message) from None
                raise
            return descriptor
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    def _register_instance(self) -> None:
        """Hold this runtime's lock before its ID can appear on any job."""
        if self.instances.is_symlink():
            raise ValueError("Research instance directory must not be a symlink.")
        self.instances.mkdir(mode=0o700, exist_ok=True)
        for _ in range(3):
            instance_id = uuid4().hex
            final = self.instances / f"{instance_id}.lock"
            # POSIX publishes a locked file by rename, so no checker can ever see
            # the final name unlocked. Windows cannot rename or delete an open
            # file, so the final name is created directly and locked at once.
            created = (
                final
                if sys.platform == "win32"
                else self.instances / f".pending-{instance_id}"
            )
            descriptor = self._private_file(created, os.O_CREAT | os.O_EXCL)
            try:
                os.write(descriptor, f"{os.getpid()}\n".encode())
                if not _try_lock(descriptor):
                    raise FileNotFoundError(created)
                if created != final:
                    os.rename(created, final)
            except FileNotFoundError:
                os.close(descriptor)
                created.unlink(missing_ok=True)
                continue
            except BaseException:
                os.close(descriptor)
                created.unlink(missing_ok=True)
                raise
            self.instance_id = instance_id
            self._instance_lock = descriptor
            self._instance_path = final
            return
        raise CapacityError("Could not register this research runtime; retry.")

    def owner_alive(self, instance_id: str) -> bool:
        """Whether the runtime that holds instance_id's lock is still running.

        A missing lock file means the owner exited. Any other failure to check is
        an explicit OwnerUnknown, never treated as a dead owner.
        """
        if instance_id == self.instance_id:
            return True
        if not _ID.fullmatch(instance_id):
            raise OwnerUnknown()
        try:
            descriptor = os.open(
                self.instances / f"{instance_id}.lock",
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            )
        except FileNotFoundError:
            return False
        except OSError:
            raise OwnerUnknown() from None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OwnerUnknown()
            try:
                return not _try_lock(descriptor)
            except OSError:
                raise OwnerUnknown() from None
        finally:
            os.close(descriptor)

    def job_owner(self, job: JobRecord) -> str:
        owner = job.get("owner")
        return owner["instance_id"] if owner else ""

    def running_elsewhere(self, job: JobRecord) -> bool:
        """A running job whose live owner is another runtime (unknown counts as live)."""
        if job["status"] != "running":
            return False
        owner = self.job_owner(job)
        if not owner or owner == self.instance_id:
            return False
        try:
            return self.owner_alive(owner)
        except OwnerUnknown:
            return True

    def _remove_stale_instance_locks(self) -> None:
        for path in self.instances.iterdir():
            name = path.name
            pending = name.startswith(".pending-")
            if path == self._instance_path or not (pending or name.endswith(".lock")):
                continue
            try:
                if (
                    pending
                    and time.time() - path.lstat().st_mtime < self.STALE_PENDING_SECONDS
                ):
                    continue
                descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            except OSError:
                continue
            try:
                if not _try_lock(descriptor):
                    continue
                if sys.platform != "win32":
                    held, current = os.fstat(descriptor), path.lstat()
                    if (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino):
                        path.unlink()
            except OSError:
                continue
            finally:
                os.close(descriptor)
            if sys.platform == "win32":
                try:
                    path.unlink()
                except OSError:
                    pass

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        # Context managers on sqlite3.Connection commit but do not close.
        connection = sqlite3.connect(self.path, timeout=self.BUSY_TIMEOUT)
        try:
            connection.execute(f"PRAGMA max_page_count={self.MAX_PAGES}")
            with connection:
                yield connection
        except sqlite3.OperationalError as exc:
            self._raise_busy(exc)
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """An immediate write transaction: reads inside it cannot go stale."""
        connection = sqlite3.connect(
            self.path, timeout=self.BUSY_TIMEOUT, isolation_level=None
        )
        try:
            connection.execute(f"PRAGMA max_page_count={self.MAX_PAGES}")
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                # SQLite may already have rolled back (for example on a full disk).
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            self._raise_busy(exc)
        finally:
            connection.close()

    @staticmethod
    def _raise_busy(exc: sqlite3.OperationalError) -> NoReturn:
        if "locked" in str(exc) or "busy" in str(exc):
            raise StoreBusy(
                "The research store is busy in another session; nothing was changed. Retry shortly."
            ) from None
        raise exc

    def create(self, payload: JobPayload) -> JobRecord:
        job: JobRecord = {**payload, "job_id": uuid4().hex, "revision": 0}
        encoded = self._encode(job)
        with self.transaction() as connection:
            count = connection.execute("SELECT count(*) FROM jobs").fetchone()[0]
            if count >= self.MAX_JOBS:
                raise HistoryFull(
                    "Research history is full. Delete a finished job before starting another."
                )
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?)",
                (job["job_id"], encoded, 0, time.time()),
            )
        return job

    def get(self, job_id: str) -> JobRecord:
        if not _ID.fullmatch(job_id):
            raise ValueError("Invalid research job ID.")
        with self.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise ValueError("Research job not found.")
        return cast(JobRecord, json.loads(row[0]))

    def list_jobs(self) -> list[JobRecord]:
        with self.connection() as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT payload FROM jobs ORDER BY updated DESC"
                )
            ]

    def summary(self) -> dict[str, Any]:
        states = {
            "awaiting_input",
            "running",
            "interrupted",
            "completed",
            "cancelled",
            "failed",
            "budget_exhausted",
        }
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT json_extract(payload, '$.status'), count(*) FROM jobs GROUP BY 1"
            )
            counts = {
                state if state in states else "unknown": count for state, count in rows
            }
            partial = connection.execute(
                "SELECT count(*) FROM jobs WHERE json_extract(payload, '$.report.status')='partial'"
            ).fetchone()[0]
        return {
            "states": counts,
            "partial_reports": partial,
            "limit": self.MAX_JOBS,
            "database_bytes": self.path.stat().st_size,
        }

    def _write(
        self, connection: sqlite3.Connection, job: JobRecord, encoded: str
    ) -> None:
        cursor = connection.execute(
            "UPDATE jobs SET payload=?, revision=?, updated=? WHERE id=? AND revision=?",
            (encoded, job["revision"] + 1, time.time(), job["job_id"], job["revision"]),
        )
        if cursor.rowcount != 1:
            if not connection.execute(
                "SELECT 1 FROM jobs WHERE id=?", (job["job_id"],)
            ).fetchone():
                raise ValueError("Research job not found.")
            raise RevisionConflict()

    def save(self, job: JobRecord, *, recovery: bool = False) -> None:
        """Conditionally write the next revision; only running jobs keep an owner."""
        if job["status"] != "running":
            job.pop("owner", None)
        updated: JobRecord = {**job, "revision": job["revision"] + 1}
        encoded = self._encode(updated, recovery=recovery)
        with self.transaction() as connection:
            self._write(connection, job, encoded)
        job["revision"] += 1

    def claim(self, job: JobRecord) -> None:
        """Save job as running and owned by this runtime, within store capacity."""
        owner: JobOwner = {"instance_id": self.instance_id, "claimed_at": time.time()}
        updated: JobRecord = {
            **job,
            "status": "running",
            "owner": owner,
            "revision": job["revision"] + 1,
        }
        encoded = self._encode(updated)
        with self.transaction() as connection:
            active = 0
            for (payload,) in connection.execute(
                "SELECT payload FROM jobs WHERE id != ? "
                "AND json_extract(payload, '$.status') = 'running'",
                (job["job_id"],),
            ):
                other = self.job_owner(json.loads(payload))
                try:
                    active += bool(other) and self.owner_alive(other)
                except OwnerUnknown:
                    active += 1
            if active >= self.MAX_RUNNING:
                raise WorkerCapacity(
                    "Research worker capacity reached across sessions; wait for a running job to finish."
                )
            self._write(connection, job, encoded)
        job["status"] = "running"
        job["owner"] = owner
        job["revision"] += 1

    def recover(
        self, job: JobRecord, *, owner_exited: bool = False
    ) -> JobRecord | None:
        """Interrupt a running job whose owner exited; None while the owner lives.

        Concurrent recoveries race on the revision: the loser reloads the winner's
        result instead of recovering twice. owner_exited lets this runtime release
        its own stale claim.
        """
        if job["status"] != "running":
            return job
        owner = self.job_owner(job)
        if owner and not owner_exited and self.owner_alive(owner):
            return None
        job["status"] = "interrupted"
        for attempt in job.get("notebook", {}).get("attempts", []):
            if attempt["status"] == "running":
                attempt["status"] = "failed"
        job["events"] = [
            *job["events"][-79:],
            {
                "message": "Runtime stopped; explicit continuation required.",
                "time": time.time(),
            },
        ]
        try:
            self.save(job, recovery=True)
        except RevisionConflict:
            latest = self.get(job["job_id"])
            return None if self.running_elsewhere(latest) else latest
        return job

    def delete(self, job_id: str) -> None:
        if not _ID.fullmatch(job_id):
            raise ValueError("Invalid research job ID.")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT json_extract(payload, '$.status') FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Research job not found.")
            if row[0] == "running":
                raise ValueError("Cancel a running job before deleting it.")
            connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def _encode(self, job: JobRecord, *, recovery: bool = False) -> str:
        encoded = json.dumps(job, ensure_ascii=False, allow_nan=False)
        limit = self.MAX_PAYLOAD_BYTES + (
            self.RECOVERY_RESERVE_BYTES if recovery else 0
        )
        if len(encoded.encode("utf-8")) > limit:
            raise CheckpointTooLarge(
                "Research checkpoint exceeds the 1 MiB storage limit."
            )
        return encoded

    def close(self) -> None:
        if self._instance_lock is not None:
            # POSIX: unlink while still locked so the name never exists unlocked.
            if sys.platform != "win32" and self._instance_path is not None:
                self._instance_path.unlink(missing_ok=True)
            os.close(self._instance_lock)
            self._instance_lock = None
            if sys.platform == "win32" and self._instance_path is not None:
                try:
                    self._instance_path.unlink(missing_ok=True)
                except OSError:
                    pass
        if self._legacy_lock is not None:
            os.close(self._legacy_lock)
            self._legacy_lock = None
