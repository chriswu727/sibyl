"""Default research location: per-user data directory, never a silent move."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sibyl.runtime import RuntimePolicy
from sibyl.storage import ArtifactStore, default_output_root, user_data_root

ROOT = Path(__file__).resolve().parents[1]


class TestDefaultOutputRoot(unittest.TestCase):
    def test_explicit_root_wins(self):
        with mock.patch.dict(os.environ, {"SIBYL_OUTPUT_ROOT": "~/custom"}):
            self.assertEqual(default_output_root(), Path("~/custom").expanduser())
            self.assertEqual(RuntimePolicy().output_root, Path("~/custom").expanduser())

    @unittest.skipIf(sys.platform in {"win32", "darwin"}, "XDG platforms")
    def test_xdg_data_home_requires_an_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {"XDG_DATA_HOME": directory, "SIBYL_OUTPUT_ROOT": ""}
            with mock.patch.dict(os.environ, environment):
                self.assertEqual(default_output_root(), Path(directory) / "sibyl")
                self.assertFalse((Path(directory) / "sibyl").exists())
            with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "relative"}):
                self.assertEqual(
                    user_data_root(), Path.home() / ".local" / "share" / "sibyl"
                )

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_new_root_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data" / "sibyl"
            ArtifactStore(root, ".research")
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_legacy_reports_store_is_reported_not_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            legacy = work / "reports" / ".research" / "jobs.sqlite3"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"old")
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "SIBYL_OUTPUT_ROOT"
            }
            environment.update(
                HOME=str(work / "home"),
                XDG_DATA_HOME=str(work / "data"),
                LOCALAPPDATA=str(work / "data"),
                PYTHONPATH=str(ROOT),
            )
            result = subprocess.run(
                [sys.executable, "-m", "sibyl.mcp_server", "--list-tools"],
                cwd=work,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("was not moved", result.stderr)
            self.assertIn("SIBYL_OUTPUT_ROOT=", result.stderr)
            self.assertEqual(legacy.read_bytes(), b"old")
            self.assertEqual(
                sorted(p.name for p in (work / "reports").iterdir()), [".research"]
            )


if __name__ == "__main__":
    unittest.main()
