from __future__ import annotations

import os
import unittest
from urllib.parse import urlparse

from tinysearch import TinySearchConfig
from tinysearch.services.web_search_service import search


@unittest.skipUnless(
    os.environ.get("TINYSEARCH_RUN_LIVE_TESTS") == "1",
    "set TINYSEARCH_RUN_LIVE_TESTS=1 to call the live DDGS backend",
)
class LiveSearchTests(unittest.TestCase):
    def test_ddgs_returns_a_public_web_result(self) -> None:
        config = TinySearchConfig(
            search_backend="ddgs",
            search_backend_fallback=False,
            ddgs_timeout_seconds=30.0,
        )
        results = search(
            "Model Context Protocol official documentation",
            limit=3,
            config=config.to_dict(),
        )

        self.assertTrue(results, "DDGS returned no search results")
        for result in results:
            parsed = urlparse(result.url)
            self.assertIn(parsed.scheme, {"http", "https"})
            self.assertTrue(parsed.netloc)


if __name__ == "__main__":
    unittest.main()
