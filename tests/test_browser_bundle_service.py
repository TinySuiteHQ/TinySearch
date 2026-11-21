from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinysearch import doctor
from tinysearch.services import browser_bundle_service


def _browsers(root: Path, *, shell: bool, full: bool = False) -> Path:
    """Lay out a Playwright browsers folder; return full Chromium's executable path.

    Playwright reports full Chromium's path even when only the headless shell
    is installed, so the returned path exists only when ``full`` is set.
    """
    executable = root / "chromium-1255" / "chrome-linux64" / "chrome"
    if full:
        executable.parent.mkdir(parents=True)
        executable.touch()
    if shell:
        shell_dir = root / "chromium_headless_shell-1255"
        (shell_dir / "chrome-headless-shell-linux64").mkdir(parents=True)
        (shell_dir / "INSTALLATION_COMPLETE").touch()
    return executable


class HeadlessShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_shell_folder_sits_beside_full_chromium_with_the_same_revision(self) -> None:
        executable = _browsers(self.root, shell=False)
        self.assertEqual(
            browser_bundle_service.headless_shell_dir(executable),
            self.root / "chromium_headless_shell-1255",
        )

    def test_shell_folder_is_unknown_outside_a_browsers_folder(self) -> None:
        self.assertIsNone(browser_bundle_service.headless_shell_dir(Path("/usr/bin/chrome")))

    def test_installed_with_only_the_shell(self) -> None:
        executable = _browsers(self.root, shell=True)
        self.assertFalse(executable.exists())
        self.assertTrue(browser_bundle_service.headless_shell_installed(executable))

    def test_not_installed_with_only_full_chromium(self) -> None:
        executable = _browsers(self.root, shell=False, full=True)
        self.assertFalse(browser_bundle_service.headless_shell_installed(executable))

    def test_not_installed_while_the_download_is_incomplete(self) -> None:
        executable = _browsers(self.root, shell=True)
        (self.root / "chromium_headless_shell-1255" / "INSTALLATION_COMPLETE").unlink()
        self.assertFalse(browser_bundle_service.headless_shell_installed(executable))


class ChromiumReadyTests(unittest.TestCase):
    def setUp(self) -> None:
        browser_bundle_service.chromium_ready.cache_clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        browser_bundle_service.chromium_ready.cache_clear()
        self.tmp.cleanup()

    def test_ready_when_the_headless_shell_is_installed(self) -> None:
        executable = _browsers(self.root, shell=True)
        with patch.object(browser_bundle_service, "_chromium_executable", return_value=executable):
            self.assertTrue(browser_bundle_service.chromium_ready())

    def test_not_ready_with_only_full_chromium(self) -> None:
        executable = _browsers(self.root, shell=False, full=True)
        with patch.object(browser_bundle_service, "_chromium_executable", return_value=executable):
            self.assertFalse(browser_bundle_service.chromium_ready())

    def test_not_ready_when_playwright_not_installed(self) -> None:
        with patch.object(browser_bundle_service, "_chromium_executable", return_value=None):
            self.assertFalse(browser_bundle_service.chromium_ready())

    def test_result_is_cached_within_the_process(self) -> None:
        executable = _browsers(self.root, shell=True)
        with patch.object(
            browser_bundle_service, "_chromium_executable", return_value=executable
        ) as resolve:
            self.assertTrue(browser_bundle_service.chromium_ready())
            self.assertTrue(browser_bundle_service.chromium_ready())

        resolve.assert_called_once_with()


class InstallChromiumTests(unittest.TestCase):
    def test_installs_only_the_headless_shell(self) -> None:
        with patch.object(browser_bundle_service.subprocess, "run") as run:
            browser_bundle_service.install_chromium()

        args = run.call_args.args[0]
        self.assertEqual(args[-3:], ["install", "--only-shell", "chromium"])

    def test_system_deps_on_linux(self) -> None:
        with patch.object(browser_bundle_service.subprocess, "run") as run, patch.object(
            browser_bundle_service.platform, "system", return_value="Linux"
        ):
            browser_bundle_service.install_chromium(with_system_deps=True)

        args = run.call_args.args[0]
        self.assertEqual(args[-4:], ["install", "--only-shell", "--with-deps", "chromium"])


class EnsureChromiumSyncTests(unittest.TestCase):
    def test_skips_install_when_already_ready(self) -> None:
        with patch.object(
            browser_bundle_service, "chromium_ready", return_value=True
        ), patch.object(browser_bundle_service, "install_chromium") as install:
            browser_bundle_service.ensure_chromium_sync()

        install.assert_not_called()

    def test_installs_when_missing(self) -> None:
        with patch.object(
            browser_bundle_service, "chromium_ready", return_value=False
        ), patch.object(browser_bundle_service, "install_chromium") as install:
            browser_bundle_service.ensure_chromium_sync(with_system_deps=True)

        install.assert_called_once_with(with_system_deps=True)


class DoctorChromiumTests(unittest.TestCase):
    """doctor is the Docker health check, so it must pass with only the shell."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _check(self, executable: Path) -> tuple[bool, str]:
        playwright = patch("playwright.sync_api.sync_playwright").start()
        self.addCleanup(patch.stopall)
        playwright.return_value.__enter__.return_value.chromium.executable_path = str(executable)
        return doctor._check_chromium()

    def test_passes_with_only_the_headless_shell(self) -> None:
        ok, detail = self._check(_browsers(self.root, shell=True))
        self.assertTrue(ok)
        self.assertEqual(detail, str(self.root / "chromium_headless_shell-1255"))

    def test_fails_with_only_full_chromium(self) -> None:
        ok, detail = self._check(_browsers(self.root, shell=False, full=True))
        self.assertFalse(ok)
        self.assertIn("tinysearch setup", detail)


if __name__ == "__main__":
    unittest.main()
