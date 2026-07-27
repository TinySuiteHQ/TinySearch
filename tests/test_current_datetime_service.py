from __future__ import annotations

import re
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from tinysearch.services.current_datetime_service import current_datetime_payload


class CurrentDatetimeServiceTests(unittest.TestCase):
    def test_payload_shape(self) -> None:
        payload = current_datetime_payload()
        self.assertEqual(set(payload), {"date_utc", "time_utc"})

    def test_payload_uses_iso_formats(self) -> None:
        fixed = datetime(2026, 6, 28, 8, 10, 0, tzinfo=UTC)
        with patch("tinysearch.services.current_datetime_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed
            payload = current_datetime_payload()

        self.assertEqual(payload["date_utc"], "2026-06-28")
        self.assertEqual(payload["time_utc"], "08:10:00")
        self.assertRegex(payload["date_utc"], re.compile(r"^\d{4}-\d{2}-\d{2}$"))
        self.assertRegex(payload["time_utc"], re.compile(r"^\d{2}:\d{2}:\d{2}$"))


if __name__ == "__main__":
    unittest.main()
