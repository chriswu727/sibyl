"""Bounded local SQLite checkpoints with one runtime owner per directory."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from .async_cache import CapacityError
from .storage import ArtifactStore


class JobStore:
    MAX_JOBS = 64
    MAX_PAYLOAD_BYTES = 1024 * 1024
    MAX_PAGES = 16384

    def __init__(self, root: Path):
        directory = ArtifactStore(root, ".research").root
        self._lock = self._private_file(directory / "owner.lock")
        try:
            if os.name == "nt":
                import msvcrt
                os.write(self._lock, b"0")
                os.lseek(self._lock, 0, os.SEEK_SET)
                msvcrt.locking(self._lock, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._lock)
            raise CapacityError("Another research runtime owns this output directory.") from None
        self.path = directory / "jobs.sqlite3"
        try:
            descriptor = self._private_file(self.path)
            os.close(descriptor)
            with self.connection() as connection:
                connection.execute("PRAGMA auto_vacuum=FULL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS jobs "
                    "(id TEXT PRIMARY KEY, payload TEXT NOT NULL, revision INTEGER NOT NULL, updated REAL NOT NULL)"
                )
            for job in self.list_jobs():
                if job["status"] == "running":
                    job["status"] = "interrupted"
                    job["events"].append({"message": "Runtime stopped; explicit continuation required.", "time": time.time()})
                    self.save(job)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _private_file(path):
        if path.is_symlink():
            raise ValueError("Research state files must not be symlinks.")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.fchmod(descriptor, 0o600)
        return descriptor

    def connection(self):
        # Context managers on sqlite3.Connection commit but do not close.
        from contextlib import contextmanager

        @contextmanager
        def connect():
            connection = sqlite3.connect(self.path, timeout=2)
            try:
                connection.execute(f"PRAGMA max_page_count={self.MAX_PAGES}")
                with connection:
                    yield connection
            finally:
                connection.close()
        return connect()

    def create(self, payload: dict) -> dict:
        with self.connection() as connection:
            count = connection.execute("SELECT count(*) FROM jobs").fetchone()[0]
            if count >= self.MAX_JOBS:
                raise CapacityError("Research history is full. Delete a finished job before starting another.")
            job = dict(payload, job_id=uuid4().hex, revision=0)
            encoded = self._encode(job)
            connection.execute("INSERT INTO jobs VALUES (?, ?, ?, ?)",
                               (job["job_id"], encoded, 0, time.time()))
        return job

    def get(self, job_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid research job ID.")
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Research job not found.")
        return json.loads(row[0])

    def list_jobs(self) -> list[dict]:
        with self.connection() as connection:
            return [json.loads(row[0]) for row in
                    connection.execute("SELECT payload FROM jobs ORDER BY updated DESC")]

    def save(self, job: dict) -> None:
        previous = job["revision"]
        updated = dict(job, revision=previous + 1)
        encoded = self._encode(updated)
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET payload=?, revision=?, updated=? WHERE id=? AND revision=?",
                (encoded, previous + 1, time.time(), job["job_id"], previous),
            )
            if cursor.rowcount != 1:
                raise ValueError("Research revision changed; reload the job.")
        job["revision"] = previous + 1

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job["status"] == "running":
            raise ValueError("Cancel a running job before deleting it.")
        with self.connection() as connection:
            connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def _encode(self, job):
        encoded = json.dumps(job, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > self.MAX_PAYLOAD_BYTES:
            raise CapacityError("Research checkpoint exceeds the 1 MiB storage limit.")
        return encoded

    def close(self):
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None
