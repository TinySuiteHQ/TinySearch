from __future__ import annotations

import unittest

from tinysearch.services.site_crawl_service import (
    BOILERPLATE_EXCLUDED_TAGS,
    _crawler_config_for_fit_markdown,
    _lightweight_browser_config,
)


class _RecordingBrowserConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class BrowserBackendConfigTests(unittest.TestCase):
    def test_default_uses_bundled_lightweight_browser(self) -> None:
        config = _lightweight_browser_config(_RecordingBrowserConfig)

        self.assertTrue(config.kwargs["light_mode"])
        self.assertTrue(config.kwargs["memory_saving_mode"])
        self.assertNotIn("cdp_url", config.kwargs)

    def test_external_cdp_preserves_backend_identity(self) -> None:
        config = _lightweight_browser_config(
            _RecordingBrowserConfig,
            {"browser_cdp_url": "http://browser:9222"},
        )

        self.assertEqual(config.kwargs["browser_mode"], "custom")
        self.assertEqual(config.kwargs["cdp_url"], "http://browser:9222")
        self.assertTrue(config.kwargs["cdp_cleanup_on_close"])
        self.assertEqual(config.kwargs["cdp_close_delay"], 0)
        self.assertIsNone(config.user_agent)
        self.assertEqual(config.browser_hint, "")


class CrawlerConfigBoilerplateExclusionTests(unittest.TestCase):
    """nav/header/footer/aside chrome must be excluded before markdown
    generation or content filtering runs in every fit_markdown_mode branch --
    a BM25 relevance threshold alone does not reliably filter this content
    out (verified empirically: identical leaked output across thresholds
    1.5-4.0 on a real site), so it has to be stripped at the DOM level."""

    def test_off_mode_excludes_boilerplate_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="off",
            user_query=None,
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)

    def test_bm25_mode_excludes_boilerplate_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="bm25",
            user_query="what is a bloom filter used for",
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)

    def test_bm25_mode_with_empty_query_falls_back_but_still_excludes_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="bm25",
            user_query="   ",
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)

    def test_pruning_mode_excludes_boilerplate_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="pruning",
            user_query=None,
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)


if __name__ == "__main__":
    unittest.main()
