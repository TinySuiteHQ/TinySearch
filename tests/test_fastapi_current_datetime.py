from __future__ import annotations

import unittest
from unittest.mock import patch

from tinysearch.servers.fastapi_server import current_datetime_endpoint


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


if __name__ == "__main__":
    unittest.main()
