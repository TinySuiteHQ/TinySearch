from __future__ import annotations

import unittest

from tinysearch.services.scrape_service import (
    _extract_metadata,
    _extract_title,
    _scan_html_meta,
)


class ScrapeHtmlMetaParserTests(unittest.TestCase):
    def test_extracts_meta_name_before_content(self) -> None:
        html = (
            '<html><head><meta name="description" content="A short description">'
            '<meta property="og:description" content="OG description"></head></html>'
        )
        meta = _scan_html_meta(html)
        self.assertEqual(meta["description"], "A short description")
        self.assertEqual(meta["og:description"], "OG description")

    def test_extracts_meta_content_before_name(self) -> None:
        html = (
            '<html><head><meta content="Alice" name="author">'
            '<meta content="2026-01-01" property="article:published_time"></head></html>'
        )
        meta = _scan_html_meta(html)
        self.assertEqual(meta["author"], "Alice")
        self.assertEqual(meta["article:published_time"], "2026-01-01")

    def test_extract_title_from_html_when_metadata_missing(self) -> None:
        html = "<html><head><title>Example Article</title></head></html>"
        title = _extract_title({}, html)
        self.assertEqual(title, "Example Article")

    def test_extract_metadata_prefers_crawl_metadata_then_html(self) -> None:
        html = '<html><head><meta name="description" content="From HTML"></head></html>'
        metadata = _extract_metadata({"title": "Crawl Title"}, html)
        self.assertEqual(metadata["description"], "From HTML")

        metadata_with_crawl = _extract_metadata(
            {"description": "From crawl", "author": "Bob"},
            html,
        )
        self.assertEqual(metadata_with_crawl["description"], "From crawl")
        self.assertEqual(metadata_with_crawl["author"], "Bob")


if __name__ == "__main__":
    unittest.main()
