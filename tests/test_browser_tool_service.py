from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tinysearch.config import normalize_config
from tinysearch.services import browser_tool_service as bt


class ToolSurfaceTests(unittest.TestCase):
    def test_exposes_exactly_nine_tools(self) -> None:
        self.assertEqual(len(bt.TOOL_NAMES), 9)

    def test_no_code_execution_or_side_effecting_tools(self) -> None:
        for banned in ("evaluate", "run_code", "fill_form", "file_upload", "drag", "drop"):
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

    async def test_snapshot_requests_ai_mode(self) -> None:
        session = self._session()
        session._page.aria_snapshot = AsyncMock(return_value="- body [ref=e1]")

        await session.snapshot()

        self.assertEqual(session._page.aria_snapshot.await_args.kwargs["mode"], "ai")

    async def test_configured_depth_is_applied(self) -> None:
        session = self._session(browser_snapshot_depth=4)
        session._page.aria_snapshot = AsyncMock(return_value="- body")

        await session.snapshot()

        self.assertEqual(session._page.aria_snapshot.await_args.kwargs["depth"], 4)

    async def test_find_returns_only_matching_context_not_the_whole_tree(self) -> None:
        session = self._session()
        tree = "\n".join(f"- generic [ref=e{i}]: row {i}" for i in range(40))
        tree += '\n- button "Accept all" [ref=e79]'
        session._page.aria_snapshot = AsyncMock(return_value=tree)

        result = await session.find(text="Accept all")

        self.assertIn("[ref=e79]", result)
        self.assertLess(len(result), len(tree))

    async def test_find_requires_exactly_one_of_text_or_regex(self) -> None:
        session = self._session()
        for kwargs in ({}, {"text": "a", "regex": "a"}):
            with self.assertRaises(bt.BrowserToolError):
                await session.find(**kwargs)

    async def test_wait_for_requires_exactly_one_condition(self) -> None:
        session = self._session()
        for kwargs in ({}, {"time_seconds": 1, "text": "a"}):
            with self.assertRaises(bt.BrowserToolError):
                await session.wait_for(**kwargs)


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

    def test_all_nine_browser_tools_ship_by_default(self) -> None:
        self.assertEqual(
            {n for n in self._names() if n.startswith("browser_")},
            {f"browser_{name}" for name in bt.TOOL_NAMES},
        )

    def test_disabling_removes_them_from_the_schema(self) -> None:
        with patch.object(
            self.mcp_server, "load_tinysearch_config", return_value={"browser_backend": "off"}
        ):
            removed = self.mcp_server.unregister_browser_tools_if_disabled()

        self.assertEqual(len(removed), 9)
        self.assertFalse({n for n in self._names() if n.startswith("browser_")})
        self.assertLessEqual({"search", "scrape_urls", "get_current_datetime"}, self._names())

    def test_enabled_backend_removes_nothing(self) -> None:
        with patch.object(
            self.mcp_server, "load_tinysearch_config", return_value={"browser_backend": "playwright"}
        ):
            self.assertEqual(self.mcp_server.unregister_browser_tools_if_disabled(), [])


if __name__ == "__main__":
    unittest.main()
