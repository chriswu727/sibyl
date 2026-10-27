"""Private, atomic artifacts within an operator-selected local root."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from .async_cache import CapacityError

LEGACY_OUTPUT_ROOT = Path("reports")


def user_data_root() -> Path:
    """The platform's per-user application data directory for Sibyl."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", "")
        parent = (
            Path(base) if os.path.isabs(base) else Path.home() / "AppData" / "Local"
        )
    elif sys.platform == "darwin":
        parent = Path.home() / "Library" / "Application Support"
    else:
        # XDG requires absolute paths; relative values are ignored.
        base = os.environ.get("XDG_DATA_HOME", "")
        parent = Path(base) if os.path.isabs(base) else Path.home() / ".local" / "share"
    return parent / "sibyl"


def default_output_root() -> Path:
    """SIBYL_OUTPUT_ROOT when set, otherwise the per-user data directory.

    Nothing is created, moved or merged here; an older ./reports store stays put.
    """
    configured = os.environ.get("SIBYL_OUTPUT_ROOT", "")
    if configured:
        return Path(configured).expanduser().absolute()
    return user_data_root()


class ArtifactStore:
    MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
    MAX_DIRECTORY_BYTES = 128 * 1024 * 1024
    MAX_ARTIFACTS = 256

    def __init__(self, root: Path | str, subdirectory: str = "."):
        base = Path(root).expanduser().resolve()
        # A newly created root is private; existing directories keep their mode.
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        relative = Path(subdirectory)
        # A Windows rooted ("\\tmp") or drive-relative ("C:x") path is not absolute
        # but would leave the root. Reject both forms on every platform alike.
        windows_anchor = PureWindowsPath(subdirectory).anchor
        if relative.anchor or windows_anchor or ".." in relative.parts:
            raise ValueError("Output directory must be relative to SIBYL_OUTPUT_ROOT.")
        target = base / relative
        for component in (target, *target.parents):
            if component == base:
                break
            if component.is_symlink():
                raise ValueError("Artifact directories must not contain symlinks.")
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = target.resolve()

    @contextmanager
    def staging(self, suffix: str):
        fd, name = tempfile.mkstemp(prefix=".sibyl-", suffix=suffix, dir=self.root)
        os.close(fd)
        temporary = Path(name)
        try:
            yield temporary
        finally:
            temporary.unlink(missing_ok=True)

    def publish(self, temporary: Path, label: str, suffix: str) -> Path:
        if (
            temporary.parent != self.root
            or temporary.is_symlink()
            or not temporary.is_file()
        ):
            raise ValueError("Artifact staging file is outside the store.")
        if not label.isalnum() or suffix not in {".md", ".pdf", ".png", ".json"}:
            raise ValueError("Invalid artifact name.")
        with self._publication_lock():
            size = temporary.stat().st_size
            paths = list(self.root.glob("sibyl_*"))
            total = sum(
                path.stat().st_size
                for path in paths
                if path.is_file() and not path.is_symlink()
            )
            if (
                size > self.MAX_ARTIFACT_BYTES
                or total + size > self.MAX_DIRECTORY_BYTES
                or len(paths) >= self.MAX_ARTIFACTS
            ):
                raise CapacityError(
                    "Artifact storage limit reached; archive or remove existing exports explicitly."
                )
            os.chmod(temporary, 0o600)
            destination = self.root / f"sibyl_{label}_{uuid4().hex}{suffix}"
            os.replace(temporary, destination)
            return destination

    @contextmanager
    def _publication_lock(self):
        path = self.root / ".artifact.lock"
        if path.is_symlink():
            raise ValueError("Artifact lock must not be a symlink.")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)

    def write(self, content: bytes, label: str, suffix: str) -> Path:
        if len(content) > self.MAX_ARTIFACT_BYTES:
            raise CapacityError("Artifact exceeds the 16 MiB export limit.")
        with self.staging(suffix) as temporary:
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            return self.publish(temporary, label, suffix)
