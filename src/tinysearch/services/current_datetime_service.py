from __future__ import annotations

from datetime import UTC, datetime


def current_datetime_payload() -> dict[str, str]:
    now = datetime.now(UTC).replace(microsecond=0)
    return {
        "date_utc": now.date().isoformat(),
        "time_utc": now.strftime("%H:%M:%S"),
    }
