"""Private, atomic artifacts within an operator-selected local root."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class ArtifactStore:
    def __init__(self, root: Path | str, subdirectory: str = "."):
        base = Path(root).expanduser().resolve()
        relative = Path(subdirectory)
        if relative.is_absolute() or ".." in relative.parts:
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
        if temporary.parent != self.root:
            raise ValueError("Artifact staging file is outside the store.")
        if not label.isalnum() or suffix not in {".md", ".pdf", ".png"}:
            raise ValueError("Invalid artifact name.")
        os.chmod(temporary, 0o600)
        destination = self.root / f"sibyl_{label}_{uuid4().hex}{suffix}"
        os.replace(temporary, destination)
        return destination

    def write(self, content: bytes, label: str, suffix: str) -> Path:
        with self.staging(suffix) as temporary:
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            return self.publish(temporary, label, suffix)
