"""Activity and saved cookies are owner-only on systems with file modes."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinysearch.paths import private_file
from tinysearch.services import activity_service as activity

POSIX = os.name != "nt"


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class PrivateFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def test_creates_the_file_and_its_folder(self) -> None:
        state = self.root / "browser" / "state.json"
        private_file(state)
        private_file(state)  # idempotent
        self.assertTrue(state.is_file())

    @unittest.skipUnless(POSIX, "file modes are POSIX; AppData is already per-user on Windows")
    def test_new_files_and_folders_are_owner_only(self) -> None:
        state = self.root / "browser" / "state.json"
        private_file(state)
        self.assertEqual(mode(state), 0o600)
        self.assertEqual(mode(state.parent), 0o700)

    @unittest.skipUnless(POSIX, "file modes are POSIX; AppData is already per-user on Windows")
    def test_an_existing_readable_file_is_tightened_but_a_chosen_folder_is_left(self) -> None:
        folder = self.root / "shared"
        folder.mkdir(mode=0o755)
        os.chmod(folder, 0o755)
        db = folder / "activity.db"
        for path in (db, Path(f"{db}-wal")):
            path.write_bytes(b"")
            os.chmod(path, 0o644)
        private_file(db)
        self.assertEqual([mode(db), mode(Path(f"{db}-wal"))], [0o600, 0o600])
        self.assertEqual(mode(folder), 0o755)

    @unittest.skipUnless(POSIX, "file modes are POSIX; AppData is already per-user on Windows")
    def test_the_activity_store_is_owner_only(self) -> None:
        db = self.root / "data" / "activity.db"
        with patch.dict(os.environ, {"TINYSEARCH_ACTIVITY": "content", "TINYSEARCH_ACTIVITY_DB": str(db)}):
            activity.reset_store()
            self.addCleanup(activity.reset_store)
            activity.get_store()
        self.assertEqual(mode(db), 0o600)
        self.assertEqual(mode(db.parent), 0o700)


if __name__ == "__main__":
    unittest.main()
