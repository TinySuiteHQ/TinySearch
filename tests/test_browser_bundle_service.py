from __future__ import annotations

import unittest
from unittest.mock import patch

from tinysearch.services import browser_bundle_service


class ChromiumReadyTests(unittest.TestCase):
    def test_ready_when_executable_exists(self) -> None:
        with patch.object(
            browser_bundle_service, "_chromium_executable"
        ) as executable:
            executable.return_value.exists.return_value = True
            self.assertTrue(browser_bundle_service.chromium_ready())

    def test_not_ready_when_executable_missing(self) -> None:
        with patch.object(
            browser_bundle_service, "_chromium_executable"
        ) as executable:
            executable.return_value.exists.return_value = False
            self.assertFalse(browser_bundle_service.chromium_ready())

    def test_not_ready_when_playwright_not_installed(self) -> None:
        with patch.object(
            browser_bundle_service, "_chromium_executable", return_value=None
        ):
            self.assertFalse(browser_bundle_service.chromium_ready())


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


if __name__ == "__main__":
    unittest.main()
