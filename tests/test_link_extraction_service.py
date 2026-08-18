from __future__ import annotations

import unittest

from tinysearch.services.link_extraction_service import (
    extract_links_from_html,
    sanitize_and_dedupe_links,
)


class ExtractLinksFromHtmlTests(unittest.TestCase):
    def test_captures_link_text_and_preceding_context(self) -> None:
        html = (
            "<html><body>"
            "<p>Read our guide on async Python.</p>"
            '<p>See the <a href="/docs/tasks">task docs</a> for details.</p>'
            "</body></html>"
        )
        links = extract_links_from_html(html, "https://example.com/")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["href"], "/docs/tasks")
        self.assertEqual(links[0]["text"], "task docs")
        self.assertIn("async Python", links[0]["context"])

    def test_skips_script_and_style_text(self) -> None:
        html = (
            "<html><body>"
            "<script>var a = '<a href=\"/fake\">nope</a>';</script>"
            "<style>a { color: red; }</style>"
            '<a href="/real">real link</a>'
            "</body></html>"
        )
        links = extract_links_from_html(html, "https://example.com/")
        self.assertEqual([link["href"] for link in links], ["/real"])

    def test_ignores_anchors_without_href(self) -> None:
        html = '<a name="top">anchor target</a><a href="/ok">ok</a>'
        links = extract_links_from_html(html, "https://example.com/")
        self.assertEqual([link["href"] for link in links], ["/ok"])

    def test_malformed_html_returns_empty_list(self) -> None:
        self.assertEqual(extract_links_from_html("<a href", "https://example.com/"), [])


class SanitizeAndDedupeLinksTests(unittest.TestCase):
    def _links(self, *hrefs: str) -> list[dict[str, str]]:
        return [{"href": href, "text": href, "context": ""} for href in hrefs]

    def test_resolves_relative_links_against_base_url(self) -> None:
        result = sanitize_and_dedupe_links(
            self._links("/next-page"),
            base_url="https://example.com/articles/one",
            blocked_domains=[],
        )
        self.assertEqual(result[0]["url"], "https://example.com/next-page")

    def test_rejects_same_page_fragment_links(self) -> None:
        result = sanitize_and_dedupe_links(
            self._links("#section-2", "https://example.com/page#section-3"),
            base_url="https://example.com/page",
            blocked_domains=[],
        )
        self.assertEqual(result, [])

    def test_strips_fragment_but_keeps_other_page_links(self) -> None:
        result = sanitize_and_dedupe_links(
            self._links("https://example.com/other#section"),
            base_url="https://example.com/page",
            blocked_domains=[],
        )
        self.assertEqual(result[0]["url"], "https://example.com/other")

    def test_rejects_unsupported_schemes(self) -> None:
        result = sanitize_and_dedupe_links(
            self._links(
                "mailto:someone@example.com",
                "javascript:void(0)",
                "tel:+15551234567",
                "ftp://example.com/file",
            ),
            base_url="https://example.com/page",
            blocked_domains=[],
        )
        self.assertEqual(result, [])

    def test_rejects_blocked_domains(self) -> None:
        result = sanitize_and_dedupe_links(
            self._links("https://blocked.example/x"),
            base_url="https://example.com/page",
            blocked_domains=["blocked.example"],
        )
        self.assertEqual(result, [])

    def test_deduplicates_canonicalized_targets(self) -> None:
        result = sanitize_and_dedupe_links(
            self._links(
                "https://example.com/next",
                "https://example.com/next#top",
                "https://Example.com/next",
            ),
            base_url="https://example.com/page",
            blocked_domains=[],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["url"], "https://example.com/next")

    def test_clips_long_card_style_link_text_to_a_label(self) -> None:
        long_text = "Release Notes " + ("word " * 60)
        result = sanitize_and_dedupe_links(
            [{"href": "/release", "text": long_text, "context": ""}],
            base_url="https://example.com/page",
            blocked_domains=[],
        )
        self.assertLessEqual(len(result[0]["text"]), 161)
        self.assertTrue(result[0]["text"].endswith("…"))
        self.assertTrue(long_text.strip().startswith(result[0]["text"].rstrip("…").strip()))

    def test_skips_empty_href(self) -> None:
        result = sanitize_and_dedupe_links(
            self._links(""),
            base_url="https://example.com/page",
            blocked_domains=[],
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
