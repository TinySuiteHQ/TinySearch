from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from tinysearch.servers.fastapi_server import (
    current_datetime_endpoint,
    get_config_endpoint,
    put_config_endpoint,
)


class FastApiCurrentDatetimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_current_datetime_payload(self) -> None:
        with patch(
            "tinysearch.core.get_current_datetime",
            return_value={
                "date_utc": "2026-06-28",
                "time_utc": "08:10:00",
            },
        ):
            payload = await current_datetime_endpoint()

        self.assertEqual(payload["date_utc"], "2026-06-28")
        self.assertEqual(payload["time_utc"], "08:10:00")

    async def test_config_endpoint_redacts_external_cdp_url(self) -> None:
        with patch(
            "tinysearch.servers.fastapi_server.load_tinysearch_config",
            return_value={
                "browser_cdp_url": "wss://example.test/cdp?token=secret",
                "search_backend": "ddgs",
            },
        ):
            payload = await get_config_endpoint()

        self.assertEqual(payload["browser_cdp_url"], "***")
        self.assertEqual(payload["search_backend"], "ddgs")

    async def test_config_endpoint_keeps_empty_cdp_default_visible(self) -> None:
        with patch(
            "tinysearch.servers.fastapi_server.load_tinysearch_config",
            return_value={"browser_cdp_url": ""},
        ):
            payload = await get_config_endpoint()

        self.assertEqual(payload["browser_cdp_url"], "")

    async def test_config_update_rejects_operator_managed_cdp_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TINYSEARCH_CONFIG_WRITABLE": "1",
                "TINYSEARCH_CONFIG_PATH": "config.json",
            },
            clear=True,
        ), patch(
            "tinysearch.servers.fastapi_server.save_tinysearch_config"
        ) as save_config:
            with self.assertRaises(HTTPException) as raised:
                await put_config_endpoint(
                    {"browser_cdp_url": "http://browser:9222"}
                )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(
            raised.exception.detail["code"], "operator_managed_config"
        )
        self.assertEqual(
            raised.exception.detail["fields"], ["browser_cdp_url"]
        )
        self.assertIn(
            "TINYSEARCH_BROWSER_CDP_URL",
            raised.exception.detail["message"],
        )
        save_config.assert_not_called()

    async def test_config_update_allows_other_fields(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TINYSEARCH_CONFIG_WRITABLE": "1",
                "TINYSEARCH_CONFIG_PATH": "config.json",
            },
            clear=True,
        ), patch(
            "tinysearch.servers.fastapi_server.save_tinysearch_config",
            return_value={"search_backend": "searxng"},
        ) as save_config:
            payload = await put_config_endpoint({"search_backend": "searxng"})

        self.assertEqual(payload, {"search_backend": "searxng"})
        save_config.assert_called_once_with({"search_backend": "searxng"})


if __name__ == "__main__":
    unittest.main()
