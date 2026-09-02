from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinysearch.config import normalize_config
from tinysearch.services import playwright_mcp_service as pw


class ExposedToolAllowlistTests(unittest.TestCase):
    def test_exposes_exactly_the_nine_supported_tools(self) -> None:
        self.assertEqual(
            set(pw.exposed_tool_names()),
            {
                "browser_navigate",
                "browser_find",
                "browser_snapshot",
                "browser_click",
                "browser_type",
                "browser_wait_for",
                "browser_take_screenshot",
                "browser_tabs",
                "browser_close",
            },
        )

    def test_code_execution_tools_are_not_exposed(self) -> None:
        """These must be absent from the schema, not merely discouraged by prompt."""
        for banned in ("browser_evaluate", "browser_run_code_unsafe"):
            self.assertNotIn(banned, pw.exposed_tool_names())

    def test_side_effecting_tools_are_not_exposed(self) -> None:
        for banned in ("browser_fill_form", "browser_file_upload", "browser_drag", "browser_drop"):
            self.assertNotIn(banned, pw.exposed_tool_names())

    def test_public_names_are_prefixed(self) -> None:
        self.assertEqual(pw.public_tool_name("browser_find"), "playwright_browser_find")


class BuildLaunchArgsTests(unittest.TestCase):
    def _config(self, **overrides) -> dict:
        return normalize_config(overrides)

    def test_pins_upstream_version(self) -> None:
        args = pw.build_launch_args(self._config())
        self.assertIn(f"@playwright/mcp@{pw.PINNED_PLAYWRIGHT_MCP_VERSION}", args)

    def test_runs_isolated_and_headless_with_images_omitted(self) -> None:
        args = pw.build_launch_args(self._config())
        self.assertIn("--isolated", args)
        self.assertIn("--headless", args)
        self.assertIn("--block-service-workers", args)
        self.assertEqual(args[args.index("--image-responses") + 1], "omit")

    def test_no_capability_groups_without_a_storage_state_path(self) -> None:
        """--caps would otherwise re-add the vision/pdf/devtools/network/testing tools."""
        self.assertNotIn("--caps", pw.build_launch_args(self._config()))

    def test_storage_capability_is_the_only_one_ever_enabled(self) -> None:
        """Needed so accepted cookies survive shutdown; never exposed to a model."""
        with tempfile.TemporaryDirectory() as tmp:
            args = pw.build_launch_args(
                self._config(browser_storage_state_path=str(Path(tmp) / "state.json"))
            )
            self.assertEqual(args.count("--caps"), 1)
            self.assertEqual(args[args.index("--caps") + 1], "storage")
            self.assertNotIn("browser_storage_state", pw.exposed_tool_names())

    def test_does_not_take_a_profile_lock(self) -> None:
        """A --user-data-dir profile can only be held by one browser at a time."""
        self.assertNotIn("--user-data-dir", pw.build_launch_args(self._config()))

    def test_action_timeout_is_passed_in_milliseconds(self) -> None:
        args = pw.build_launch_args(self._config(browser_action_timeout_seconds=7.5))
        self.assertEqual(args[args.index("--timeout-action") + 1], "7500")

    def test_cdp_url_becomes_cdp_endpoint(self) -> None:
        args = pw.build_launch_args(self._config(browser_cdp_url="ws://127.0.0.1:9222/x"))
        self.assertEqual(args[args.index("--cdp-endpoint") + 1], "ws://127.0.0.1:9222/x")

    def test_omits_cdp_endpoint_when_unset(self) -> None:
        self.assertNotIn("--cdp-endpoint", pw.build_launch_args(self._config()))

    def test_storage_state_is_passed_and_its_parent_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "state.json"
            args = pw.build_launch_args(
                self._config(browser_storage_state_path=str(target))
            )
            self.assertEqual(args[args.index("--storage-state") + 1], str(target))
            self.assertTrue(target.parent.is_dir())

    def test_missing_storage_state_file_is_seeded(self) -> None:
        """Upstream refuses to start when --storage-state points at a missing file."""
        import json

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "state.json"
            pw.build_launch_args(self._config(browser_storage_state_path=str(target)))
            self.assertTrue(target.is_file())
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"cookies": [], "origins": []},
            )

    def test_existing_storage_state_file_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "state.json"
            target.write_text('{"cookies": [{"name": "kept"}], "origins": []}', encoding="utf-8")
            pw.build_launch_args(self._config(browser_storage_state_path=str(target)))
            self.assertIn("kept", target.read_text(encoding="utf-8"))


class CapResponseTests(unittest.TestCase):
    def test_short_response_passes_through_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "a snapshot"
            self.assertEqual(
                pw.cap_response(
                    text, char_budget=100, output_dir=Path(tmp), tool_name="browser_snapshot"
                ),
                text,
            )

    def test_zero_budget_disables_capping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "x" * 5000
            self.assertEqual(
                pw.cap_response(
                    text, char_budget=0, output_dir=Path(tmp), tool_name="browser_snapshot"
                ),
                text,
            )

    def test_oversized_response_is_truncated_and_spilled_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "y" * 5000
            result = pw.cap_response(
                text, char_budget=100, output_dir=Path(tmp), tool_name="browser_snapshot"
            )

            self.assertLess(len(result), len(text))
            self.assertTrue(result.startswith("y" * 100))
            self.assertIn("truncated browser_snapshot response", result)

            spilled = list(Path(tmp).glob("browser_snapshot-*.txt"))
            self.assertEqual(len(spilled), 1)
            self.assertEqual(spilled[0].read_text(encoding="utf-8"), text)
            self.assertIn(str(spilled[0]), result)

    def test_unwritable_output_dir_still_returns_capped_text(self) -> None:
        """A spill failure must not turn a successful browser call into an error."""
        result = pw.cap_response(
            "z" * 500,
            char_budget=50,
            output_dir=Path("/nonexistent") / "tinysearch" / "nope",
            tool_name="browser_find",
        )
        self.assertTrue(result.startswith("z" * 50))
        self.assertIn("unavailable", result)


class BackendGateTests(unittest.TestCase):
    def test_backend_defaults_to_off(self) -> None:
        self.assertFalse(pw.browser_backend_enabled(normalize_config()))

    def test_enabled_only_for_playwright_mcp(self) -> None:
        self.assertTrue(
            pw.browser_backend_enabled(normalize_config({"browser_backend": "playwright_mcp"}))
        )

    def test_get_client_raises_when_disabled(self) -> None:
        with self.assertRaises(pw.PlaywrightMcpDisabledError):
            pw.get_client(normalize_config())

    def test_config_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            normalize_config({"browser_backend": "selenium"})


if __name__ == "__main__":
    unittest.main()
