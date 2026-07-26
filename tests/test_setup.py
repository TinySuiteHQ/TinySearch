from __future__ import annotations

import unittest
from unittest.mock import patch

from tinysearch import setup as setup_module


class SetupRunTests(unittest.TestCase):
    def test_run_installs_chromium_and_downloads_model(self) -> None:
        config = {"embedding_backend": "onnx", "embedding_model": "fast"}
        with patch.object(setup_module, "_install_chromium") as install, patch.object(
            setup_module, "load_research_config", return_value=config
        ), patch(
            "tinysearch.services.onnx_bundle_service.ensure_onnx_bundle_sync"
        ) as ensure:
            exit_code = setup_module.run()

        install.assert_called_once_with(False)
        ensure.assert_called_once_with("fast")
        self.assertEqual(exit_code, 0)

    def test_run_skips_download_for_non_onnx_backend(self) -> None:
        config = {"embedding_backend": "openai_compatible", "embedding_model": "fast"}
        with patch.object(setup_module, "_install_chromium"), patch.object(
            setup_module, "load_research_config", return_value=config
        ), patch(
            "tinysearch.services.onnx_bundle_service.ensure_onnx_bundle_sync"
        ) as ensure:
            setup_module.run()

        ensure.assert_not_called()

    def test_with_system_deps_passed_through(self) -> None:
        config = {"embedding_backend": "openai_compatible", "embedding_model": "fast"}
        with patch.object(setup_module, "_install_chromium") as install, patch.object(
            setup_module, "load_research_config", return_value=config
        ):
            setup_module.run(with_system_deps=True)

        install.assert_called_once_with(True)


class InstallChromiumTests(unittest.TestCase):
    def test_non_linux_ignores_with_system_deps(self) -> None:
        with patch("tinysearch.setup.platform.system", return_value="Darwin"), patch(
            "tinysearch.setup.subprocess.run"
        ) as run:
            setup_module._install_chromium(with_system_deps=True)

        args = run.call_args.args[0]
        self.assertNotIn("--with-deps", args)

    def test_linux_adds_with_deps_flag(self) -> None:
        with patch("tinysearch.setup.platform.system", return_value="Linux"), patch(
            "tinysearch.setup.subprocess.run"
        ) as run:
            setup_module._install_chromium(with_system_deps=True)

        args = run.call_args.args[0]
        self.assertIn("--with-deps", args)

    def test_without_flag_never_adds_with_deps(self) -> None:
        with patch("tinysearch.setup.platform.system", return_value="Linux"), patch(
            "tinysearch.setup.subprocess.run"
        ) as run:
            setup_module._install_chromium(with_system_deps=False)

        args = run.call_args.args[0]
        self.assertNotIn("--with-deps", args)


if __name__ == "__main__":
    unittest.main()
