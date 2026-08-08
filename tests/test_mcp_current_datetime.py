from __future__ import annotations

import unittest
from inspect import signature
from unittest.mock import patch
from xml.etree import ElementTree

from mcp import types

from tinysearch.servers.mcp_server import get_current_datetime_tool, mcp
from tinysearch.servers.fastapi_server import current_datetime_endpoint


def _fn(coro):
    return getattr(coro, "fn", coro)


class McpCurrentDatetimeTests(unittest.IsolatedAsyncioTestCase):
    def test_mcp_signature_has_no_parameters(self) -> None:
        self.assertEqual(list(signature(_fn(get_current_datetime_tool)).parameters), [])

    async def test_returns_current_datetime_xml(self) -> None:
        with patch(
            "tinysearch.core.get_current_datetime",
            return_value={
                "date_utc": "2026-06-28",
                "time_utc": "08:10:00",
            },
        ):
            answer = await _fn(get_current_datetime_tool)()

        self.assertEqual(
            answer,
            "<current_datetime>\n<date_utc>\n2026-06-28\n</date_utc>\n"
            "<time_utc>\n08:10:00\n</time_utc>\n</current_datetime>",
        )
        self.assertEqual(ElementTree.fromstring(answer).tag, "current_datetime")

    async def test_mcp_datetime_wire_text_is_well_formed_xml(self) -> None:
        with patch(
            "tinysearch.core.get_current_datetime",
            return_value={"date_utc": "2026-06-28", "time_utc": "08:10:00"},
        ):
            content, structured = await mcp._tool_manager.call_tool(
                "get_current_datetime", {}, convert_result=True
            )

        self.assertEqual(len(content), 1)
        self.assertIsInstance(content[0], types.TextContent)
        root = ElementTree.fromstring(content[0].text)
        self.assertEqual(root.findtext("date_utc", default="").strip(), "2026-06-28")
        self.assertEqual(structured, {"result": content[0].text})

    async def test_fastapi_retains_its_structured_payload(self) -> None:
        with patch(
            "tinysearch.core.get_current_datetime",
            return_value={
                "date_utc": "2026-06-28",
                "time_utc": "08:10:00",
            },
        ):
            fastapi_payload = await current_datetime_endpoint()

        self.assertEqual(fastapi_payload["date_utc"], "2026-06-28")


if __name__ == "__main__":
    unittest.main()
