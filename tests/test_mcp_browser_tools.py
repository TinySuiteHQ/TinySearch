from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from tinysearch.servers import mcp_server
from tinysearch.services import playwright_mcp_service as pw

# One representative upstream schema. The proxy must re-export whatever the
# child reports rather than restating arguments in a Python signature.
_NAVIGATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"url": {"type": "string", "description": "The URL to navigate to"}},
    "required": ["url"],
    "additionalProperties": False,
}


def _fake_schemas() -> dict[str, dict]:
    return {
        name: (_NAVIGATE_SCHEMA if name == "browser_navigate" else {"type": "object", "properties": {}})
        for name in pw.exposed_tool_names()
    }


class RegisterBrowserToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._original = dict(mcp_server.mcp._tool_manager._tools)

    def tearDown(self) -> None:
        mcp_server.mcp._tool_manager._tools = dict(self._original)

    async def test_registers_nothing_when_backend_is_off(self) -> None:
        with patch.object(mcp_server, "load_tinysearch_config", return_value={"browser_backend": "off"}):
            self.assertEqual(await mcp_server.register_browser_tools(), [])

    async def _register(self) -> list[str]:
        with patch.object(
            mcp_server, "load_tinysearch_config", return_value={"browser_backend": "playwright_mcp"}
        ), patch.object(pw, "fetch_tool_schemas", AsyncMock(return_value=_fake_schemas())):
            return await mcp_server.register_browser_tools()

    async def test_registers_all_nine_allowlisted_tools(self) -> None:
        registered = await self._register()
        self.assertEqual(len(registered), 9)
        self.assertIn("playwright_browser_navigate", registered)

    async def test_upstream_schema_is_re_exported_verbatim(self) -> None:
        await self._register()
        listed = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}
        self.assertEqual(listed["playwright_browser_navigate"].inputSchema, _NAVIGATE_SCHEMA)

    async def test_code_execution_tools_never_reach_the_schema(self) -> None:
        await self._register()
        names = {tool.name for tool in await mcp_server.mcp.list_tools()}
        for banned in ("browser_evaluate", "browser_run_code_unsafe"):
            self.assertNotIn(banned, names)
            self.assertNotIn(f"playwright_{banned}", names)

    async def test_existing_research_tools_survive_registration(self) -> None:
        await self._register()
        names = {tool.name for tool in await mcp_server.mcp.list_tools()}
        self.assertLessEqual({"search", "scrape_urls", "get_current_datetime"}, names)

    async def test_call_forwards_arguments_to_the_child_unchanged(self) -> None:
        await self._register()
        client = AsyncMock()
        client.call = AsyncMock(return_value="### Page state")

        with patch.object(
            mcp_server, "load_tinysearch_config", return_value={"browser_backend": "playwright_mcp"}
        ), patch.object(pw, "get_client", return_value=client):
            result = await mcp_server.mcp.call_tool(
                "playwright_browser_navigate", {"url": "https://example.com"}
            )

        client.call.assert_awaited_once_with("browser_navigate", {"url": "https://example.com"})
        self.assertIn("Page state", str(result))


if __name__ == "__main__":
    unittest.main()
