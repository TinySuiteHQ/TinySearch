from __future__ import annotations

import unittest
from inspect import signature
from unittest.mock import patch

from tinysearch.servers.mcp_server import get_current_datetime_tool
from tinysearch.servers.fastapi_server import current_datetime_endpoint


def _fn(coro):
    return getattr(coro, "fn", coro)


class McpCurrentDatetimeTests(unittest.IsolatedAsyncioTestCase):
    def test_mcp_signature_has_no_parameters(self) -> None:
        self.assertEqual(list(signature(_fn(get_current_datetime_tool)).parameters), [])

    async def test_returns_current_datetime_payload(self) -> None:
        with patch(
            "tinysearch.core.get_current_datetime",
            return_value={
                "date_utc": "2026-06-28",
                "time_utc": "08:10:00",
            },
        ):
            payload = await _fn(get_current_datetime_tool)()

        self.assertEqual(payload["date_utc"], "2026-06-28")
        self.assertEqual(payload["time_utc"], "08:10:00")

    async def test_mcp_and_fastapi_return_same_payload(self) -> None:
        with patch(
            "tinysearch.core.get_current_datetime",
            return_value={
                "date_utc": "2026-06-28",
                "time_utc": "08:10:00",
            },
        ):
            mcp_payload = await _fn(get_current_datetime_tool)()
            fastapi_payload = await current_datetime_endpoint()

        self.assertEqual(mcp_payload, fastapi_payload)


if __name__ == "__main__":
    unittest.main()
