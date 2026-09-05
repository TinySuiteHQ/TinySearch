from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tinysearch.config import normalize_config
from tinysearch.services import browser_tool_service as bt


class ToolSurfaceTests(unittest.TestCase):
    def test_exposes_exactly_seven_operations(self) -> None:
        self.assertEqual(len(bt.TOOL_NAMES), 7)

    def test_no_code_execution_or_side_effecting_tools(self) -> None:
        for banned in (
            "evaluate",
            "run_code",
            "fill_form",
            "file_upload",
            "drag",
            "drop",
            "take_screenshot",
        ):
            self.assertNotIn(banned, bt.TOOL_NAMES)

    def test_playwright_floor_is_checked_at_runtime(self) -> None:
        """The aria_snapshot mode/depth API is recent and has changed shape before."""
        stub = MagicMock()
        stub.aria_snapshot = lambda self, *, timeout=None: None  # pre-1.59 shape
        with patch.dict("sys.modules", {"playwright.async_api": MagicMock(Locator=stub)}):
            with self.assertRaises(bt.BrowserToolError) as raised:
                bt.ensure_supported()
        self.assertIn("1.59", str(raised.exception))


class RefValidationTests(unittest.TestCase):
    def test_accepts_a_snapshot_ref(self) -> None:
        self.assertEqual(bt._validate_ref("e42"), "e42")

    def test_rejects_raw_selectors_and_injection_attempts(self) -> None:
        """A target must be a ref the model observed, never an arbitrary selector."""
        for bad in ("div.foo", "#id", "", "  ", "e1, script", 'a[href="x"]', "e1 >> nth=0"):
            with self.assertRaises(bt.BrowserToolError):
                bt._validate_ref(bad)


class CapTests(unittest.TestCase):
    def test_short_text_passes_through(self) -> None:
        self.assertEqual(bt.cap("hello", 100), "hello")

    def test_zero_budget_disables_capping(self) -> None:
        text = "x" * 5000
        self.assertEqual(bt.cap(text, 0), text)

    def test_oversized_text_is_trimmed_and_suggests_a_cheaper_call(self) -> None:
        result = bt.cap("y" * 5000, 100)
        self.assertTrue(result.startswith("y" * 100))
        self.assertIn("truncated at 100 of 5000", result)
        self.assertIn("smaller depth", result)


class BackendGateTests(unittest.TestCase):
    def test_browser_is_enabled_by_default(self) -> None:
        self.assertTrue(bt.browser_backend_enabled(normalize_config()))

    def test_off_disables_it(self) -> None:
        config = normalize_config({"browser_backend": "off"})
        self.assertFalse(bt.browser_backend_enabled(config))
        with self.assertRaises(bt.BrowserDisabledError):
            bt.get_session(config)

    def test_config_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            normalize_config({"browser_backend": "node"})


class SessionBehaviourTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, **overrides) -> bt.BrowserSession:
        session = bt.BrowserSession(normalize_config(overrides))
        session._page = MagicMock()
        session._page.is_closed.return_value = False
        session._page.url = "https://example.com"
        session._page.title = AsyncMock(return_value="T")
        session._page.wait_for_load_state = AsyncMock()
        session._page.aria_snapshot = AsyncMock(return_value="- body")
        session._context = MagicMock()
        session.start = AsyncMock()
        return session

    async def test_click_addresses_the_element_by_aria_ref(self) -> None:
        session = self._session()
        locator = MagicMock(click=AsyncMock())
        session._page.locator = MagicMock(return_value=locator)
        session._page.aria_snapshot = AsyncMock(return_value="- body")

        await session.click("e42")

        session._page.locator.assert_any_call("aria-ref=e42")
        locator.click.assert_awaited_once()

    async def test_click_refuses_a_raw_selector(self) -> None:
        session = self._session()
        with self.assertRaises(bt.BrowserToolError):
            await session.click("button.accept")

    async def test_look_requests_ai_mode(self) -> None:
        session = self._session()
        session._page.aria_snapshot = AsyncMock(return_value="- body [ref=e1]")

        await session.look()

        self.assertEqual(session._page.aria_snapshot.await_args.kwargs["mode"], "ai")

    async def test_configured_depth_is_applied(self) -> None:
        session = self._session(browser_snapshot_depth=4)
        session._page.aria_snapshot = AsyncMock(return_value="- body")

        await session.look()

        self.assertEqual(session._page.aria_snapshot.await_args.kwargs["depth"], 4)

    async def test_look_with_find_returns_only_matching_context(self) -> None:
        session = self._session()
        tree = "\n".join(f"- generic [ref=e{i}]: row {i}" for i in range(40))
        tree += '\n- button "Accept all" [ref=e79]'
        session._page.aria_snapshot = AsyncMock(return_value=tree)

        result = await session.look(find="Accept all")

        self.assertIn("[ref=e79]", result)
        self.assertLess(len(result), len(tree))

    async def test_acting_and_finding_are_one_call(self) -> None:
        """The point of the merge: a click reports back only what was asked for."""
        session = self._session()
        tree = "\n".join(f"- generic [ref=e{i}]: row {i}" for i in range(40))
        tree += '\n- heading "Results" [ref=e88]'
        session._page.aria_snapshot = AsyncMock(return_value=tree)
        session._page.locator = MagicMock(return_value=MagicMock(click=AsyncMock()))

        result = await session.click("e42", find="Results")

        self.assertIn("[ref=e88]", result)
        self.assertNotIn("row 7", result)

    async def test_no_matches_is_not_silently_a_full_tree(self) -> None:
        """An empty result is an answer about the page, not a reason to dump it."""
        session = self._session()
        session._page.aria_snapshot = AsyncMock(return_value="- body [ref=e1]: hello")

        result = await session.look(find="absent")

        self.assertIn("No matches", result)
        self.assertNotIn("[ref=e1]", result)

    async def test_find_matches_as_regex(self) -> None:
        session = self._session()
        tree = "\n".join(f"- generic [ref=e{i}]: row {i}" for i in range(40))
        tree += '\n- button "Accept all" [ref=e79]\n- button "Submit" [ref=e80]'
        session._page.aria_snapshot = AsyncMock(return_value=tree)

        result = await session.look(find="Accept all|Submit")

        self.assertIn("[ref=e79]", result)
        self.assertIn("[ref=e80]", result)

    async def test_find_falls_back_to_literal_when_not_valid_regex(self) -> None:
        """A leading '*' is invalid regex (nothing to repeat); it must still match literally."""
        session = self._session()
        tree = "\n".join(f"- generic [ref=e{i}]: row {i}" for i in range(40))
        tree += '\n- heading "*half off today" [ref=e79]'
        session._page.aria_snapshot = AsyncMock(return_value=tree)

        result = await session.look(find="*half off")

        self.assertIn("[ref=e79]", result)

    async def test_find_context_reaches_fields_nested_below_the_match(self) -> None:
        """A match on a title must not cut off the views/date nested under a
        sibling wrapper several levels below it -- the record boundary, not a
        fixed +/-2 line window, decides how much context comes back."""
        session = self._session()

        def video(n: int) -> str:
            return (
                f"  - generic [ref=v{n}] [cursor=pointer]:\n"
                f"    - link [ref=l{n}]:\n"
                f"      - /url: /watch?v=id{n}\n"
                f'      - generic [ref=d{n}]: {n}:00\n'
                f"    - generic [ref=i{n}]:\n"
                f"      - generic [ref=w{n}]:\n"
                f'        - heading "video number {n}" [ref=h{n}]\n'
                f"        - group [ref=g{n}]:\n"
                f'          - generic [ref=vc{n}]: {n}K views\n'
                f'          - generic "1 day ago" [ref=ts{n}]\n'
            )

        tree = "- generic [ref=list]:\n" + "".join(video(i) for i in range(6))
        session._page.aria_snapshot = AsyncMock(return_value=tree)

        result = await session.look(find="video number 3")

        self.assertIn("[ref=h3]", result)
        self.assertIn("3K views", result)
        self.assertIn("1 day ago", result)

    async def test_find_climbs_from_the_match_not_the_padded_window(self) -> None:
        """A regression guard: two matches inside one table row (the link's
        visible text and its /url both contain "diameter") must each expand
        to their own complete row, not a mangled or duplicated one -- the
        climb has to start from the match line itself, not from two lines
        above it, or it skips straight past the true immediate parent."""
        session = self._session()

        def row(name: str, n: int, value: str) -> str:
            return (
                f'  - row [ref=r{n}]:\n'
                f'    - rowheader [ref=rh{n}]:\n'
                f'      - link "{name}" [ref=l{n}]:\n'
                f'        - /url: https://en.wikipedia.org/wiki/{name.replace(" ", "_")}\n'
                f'    - cell [ref=c{n}]:\n'
                f'      - text: {value}\n'
            )

        fields = [
            ("Orbital period", "87.97 days"),
            ("Aphelion", "69,816,900 km"),
            ("Perihelion", "46,001,200 km"),
            ("Eccentricity", "0.205630"),
            ("Mean diameter", "4,879.4 km"),
            ("Angular diameter", "4.5-13 arcsec"),
            ("Mass", "3.3011e23 kg"),
        ]
        tree = "- table [ref=infobox]:\n" + "".join(row(n, i, v) for i, (n, v) in enumerate(fields))
        session._page.aria_snapshot = AsyncMock(return_value=tree)

        result = await session.look(find="diameter")

        self.assertEqual(result.count("[ref=c4]"), 1)
        self.assertEqual(result.count("[ref=c5]"), 1)
        self.assertIn("text: 4,879.4 km", result)
        self.assertIn("text: 4.5-13 arcsec", result)
        # Neighbouring records must not leak into a single video's match.
        self.assertNotIn("video number 2", result)
        self.assertNotIn("video number 4", result)

    async def test_wait_for_requires_exactly_one_condition(self) -> None:
        session = self._session()
        for kwargs in ({}, {"time_seconds": 1, "text": "a"}):
            with self.assertRaises(bt.BrowserToolError):
                await session.wait_for(**kwargs)

    async def test_idle_timer_closes_a_warm_browser(self) -> None:
        session = self._session(browser_idle_shutdown_seconds=0.01)
        context = session._context
        context.close = AsyncMock()
        browser = MagicMock(close=AsyncMock())
        playwright = MagicMock(stop=AsyncMock())
        session._browser = browser
        session._playwright = playwright

        session.schedule_idle_shutdown()
        await asyncio.sleep(0.03)

        self.assertFalse(session.started)
        context.close.assert_awaited_once()
        browser.close.assert_awaited_once()


class ResolveActArgumentsTests(unittest.TestCase):
    def test_every_folded_action_is_routable(self) -> None:
        self.assertEqual(set(bt.ACT_ACTIONS), set(bt.TOOL_NAMES) - {"navigate"})

    def test_view_arguments_reach_every_action_that_returns_a_view(self) -> None:
        for action in ("look", "click", "type", "wait_for", "tabs"):
            accepted, _ = bt._ACT_PARAMETERS[action]
            self.assertLessEqual(set(bt.VIEW_ARGUMENTS), set(accepted), action)

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaises(bt.BrowserToolError):
            bt.resolve_act_arguments("evaluate", {})

    def test_arguments_for_other_actions_are_dropped(self) -> None:
        """A stray argument should not fail a call it does not apply to."""
        resolved = bt.resolve_act_arguments("click", {"target": "e1", "time_seconds": 2})
        self.assertEqual(resolved, {"target": "e1"})

    def test_missing_required_argument_names_itself(self) -> None:
        with self.assertRaises(bt.BrowserToolError) as raised:
            bt.resolve_act_arguments("click", {"target": ""})
        self.assertIn("requires target", str(raised.exception))

        with self.assertRaises(bt.BrowserToolError) as raised:
            bt.resolve_act_arguments("type", {"target": "e1", "text": ""})
        self.assertIn("requires text", str(raised.exception))

    def test_empty_and_zero_sentinels_mean_absent(self) -> None:
        self.assertEqual(bt.resolve_act_arguments("look", {"depth": 0, "find": ""}), {})
        self.assertEqual(bt.resolve_act_arguments("wait_for", {"time_seconds": 0.0}), {})

    def test_false_is_a_real_boolean_not_an_absent_value(self) -> None:
        resolved = bt.resolve_act_arguments("type", {"target": "e1", "text": "x", "submit": False})
        self.assertIs(resolved["submit"], False)

    def test_tab_index_zero_is_preserved(self) -> None:
        """Tab 0 is a valid target, so zero cannot mean 'absent' here."""
        resolved = bt.resolve_act_arguments("tabs", {"action": "select", "index": 0})
        self.assertEqual(resolved, {"action": "select", "index": 0})

    def test_close_takes_no_arguments(self) -> None:
        self.assertEqual(bt.resolve_act_arguments("close", {"target": "e1"}), {})


class CallToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaises(bt.BrowserToolError):
            await bt.call_tool("evaluate", normalize_config())

    async def test_response_is_capped_using_config(self) -> None:
        session = MagicMock()
        session.navigate = AsyncMock(return_value="z" * 900)
        config = normalize_config({"browser_response_char_budget": 50})
        with patch.object(bt, "get_session", return_value=session):
            result = await bt.call_tool("navigate", config, url="https://example.com")
        self.assertIn("truncated at 50 of 900", result)


class ToolRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        from tinysearch.servers import mcp_server

        self.mcp_server = mcp_server
        self._original = dict(mcp_server.mcp._tool_manager._tools)

    def tearDown(self) -> None:
        self.mcp_server.mcp._tool_manager._tools = dict(self._original)

    def _names(self) -> set[str]:
        return set(self.mcp_server.mcp._tool_manager._tools)

    def test_browser_ships_as_two_tools_not_seven(self) -> None:
        """MCP cannot group tools, so six lifecycle actions fold into browser_act."""
        self.assertEqual(
            {n for n in self._names() if n.startswith("browser_")},
            {"browser_navigate", "browser_act"},
        )

    def test_disabling_removes_them_from_the_schema(self) -> None:
        with patch.object(
            self.mcp_server, "load_tinysearch_config", return_value={"browser_backend": "off"}
        ):
            removed = self.mcp_server.unregister_browser_tools_if_disabled()

        self.assertEqual(len(removed), 2)
        self.assertFalse({n for n in self._names() if n.startswith("browser_")})
        self.assertLessEqual({"search", "scrape_urls", "get_current_datetime"}, self._names())

    def test_enabled_backend_removes_nothing(self) -> None:
        with patch.object(
            self.mcp_server, "load_tinysearch_config", return_value={"browser_backend": "playwright"}
        ):
            self.assertEqual(self.mcp_server.unregister_browser_tools_if_disabled(), [])


if __name__ == "__main__":
    unittest.main()
