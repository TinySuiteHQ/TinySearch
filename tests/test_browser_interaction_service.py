from __future__ import annotations

import unittest

from tinysearch.services.browser_interaction_service import (
    BrowserActionFailedError,
    InvalidBrowserActionError,
    _raise_on_action_failure,
    build_action_script,
)


class BuildActionScriptTests(unittest.TestCase):
    def test_click_requires_selector(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script({"action": "click"}, default_timeout_seconds=10)

    def test_click_embeds_selector_and_timeout(self) -> None:
        script = build_action_script(
            {"action": "click", "selector": "#go", "timeout_seconds": 5},
            default_timeout_seconds=10,
        )
        self.assertIn('"#go"', script)
        self.assertIn("Date.now() + 5000", script)
        self.assertIn("el.click();", script)

    def test_type_requires_text(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script(
                {"action": "type", "selector": "#q"}, default_timeout_seconds=10
            )

    def test_type_escapes_text_safely(self) -> None:
        script = build_action_script(
            {"action": "type", "selector": "#q", "text": 'a"b\\c'},
            default_timeout_seconds=10,
        )
        # json.dumps must be the only thing responsible for embedding the
        # value; a naive f-string interpolation would break out of the
        # generated JS string literal on an unescaped quote or backslash.
        self.assertIn('"a\\"b\\\\c"', script)

    def test_type_submit_dispatches_enter_and_requests_submit(self) -> None:
        script = build_action_script(
            {"action": "type", "selector": "#q", "text": "hi", "submit": True},
            default_timeout_seconds=10,
        )
        self.assertIn("KeyboardEvent", script)
        self.assertIn("requestSubmit", script)

    def test_scroll_requires_exactly_one_target(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script({"action": "scroll"}, default_timeout_seconds=10)
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script(
                {"action": "scroll", "to": "bottom", "amount": 100},
                default_timeout_seconds=10,
            )

    def test_scroll_to_bottom(self) -> None:
        script = build_action_script(
            {"action": "scroll", "to": "bottom"}, default_timeout_seconds=10
        )
        self.assertEqual(script, "window.scrollTo(0, document.body.scrollHeight);")

    def test_scroll_invalid_to_value(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script(
                {"action": "scroll", "to": "middle"}, default_timeout_seconds=10
            )

    def test_scroll_amount_must_be_int(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script(
                {"action": "scroll", "amount": "far"}, default_timeout_seconds=10
            )

    def test_wait_requires_exactly_one_of_seconds_or_selector(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script({"action": "wait"}, default_timeout_seconds=10)
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script(
                {"action": "wait", "seconds": 1, "selector": "#x"},
                default_timeout_seconds=10,
            )

    def test_wait_seconds_must_be_positive(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script({"action": "wait", "seconds": 0}, default_timeout_seconds=10)

    def test_wait_selector_script(self) -> None:
        script = build_action_script(
            {"action": "wait", "selector": ".results", "timeout_seconds": 3},
            default_timeout_seconds=10,
        )
        self.assertIn('".results"', script)
        self.assertIn("Date.now() + 3000", script)

    def test_unknown_action_rejected(self) -> None:
        with self.assertRaises(InvalidBrowserActionError):
            build_action_script({"action": "hover", "selector": "#x"}, default_timeout_seconds=10)

    def test_scripts_are_not_self_wrapped_in_their_own_async_iife(self) -> None:
        # Crawl4AI already wraps each js_code entry in its own awaited
        # `async () => { <script> }` and catches what it throws. Wrapping the
        # body in a second, un-awaited `(async () => {...})();` here would let
        # crawl4ai's wrapper return before the action finishes and would turn
        # a thrown error into a silently dropped unhandled rejection instead
        # of a value crawl4ai's own catch turns into a checkable result --
        # exactly the regression this test guards against.
        for action in (
            {"action": "click", "selector": "#x"},
            {"action": "type", "selector": "#x", "text": "hi"},
            {"action": "wait", "seconds": 1},
            {"action": "wait", "selector": "#x"},
        ):
            script = build_action_script(action, default_timeout_seconds=10)
            self.assertNotIn("(async () => {", script)


class RaiseOnActionFailureTests(unittest.TestCase):
    def test_no_op_for_missing_or_malformed_result(self) -> None:
        _raise_on_action_failure(None)
        _raise_on_action_failure({"success": True})
        _raise_on_action_failure({"success": True, "results": "not-a-list"})

    def test_all_successful_results_do_not_raise(self) -> None:
        _raise_on_action_failure(
            {"success": True, "results": [{"success": True}, {"success": True}]}
        )

    def test_a_failed_script_raises_with_its_error_and_position(self) -> None:
        with self.assertRaises(BrowserActionFailedError) as ctx:
            _raise_on_action_failure(
                {
                    "success": True,
                    "results": [
                        {"success": True},
                        {"success": False, "error": "target not found: #missing"},
                    ],
                }
            )
        self.assertIn("action 2", str(ctx.exception))
        self.assertIn("target not found: #missing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
