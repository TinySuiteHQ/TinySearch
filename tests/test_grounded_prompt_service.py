from __future__ import annotations

import unittest
from datetime import UTC, datetime
from xml.etree import ElementTree

from tinysearch.services.grounded_prompt_service import (
    format_browse_result,
    format_relevant_text,
    format_url_grounded_prompt,
)
from tinysearch.results import result_envelope


class FormatRelevantTextTests(unittest.TestCase):
    def test_emits_one_element_per_non_empty_chunk_keeping_ordinals(self) -> None:
        out = format_relevant_text(
            [{"text": "alpha"}, {"text": ""}, {"text": "beta"}]
        )

        self.assertIn('<chunk index="1">\nalpha\n</chunk>', out)
        self.assertIn('<chunk index="3">\nbeta\n</chunk>', out)
        self.assertNotIn('index="2"', out)

    def test_returns_empty_string_when_no_chunks(self) -> None:
        self.assertEqual(format_relevant_text([]), "")


class FormatUrlGroundedPromptTests(unittest.TestCase):
    def test_emits_parseable_page_xml_with_diagnostics(self) -> None:
        prompt = format_url_grounded_prompt(
            question="What does this page say about installation?",
            url="https://example.com/article",
            title="Example Article",
            ranked_chunks=[{"text": "Run pip install."}, {"text": "Then run the CLI."}],
            today="2026-06-12",
            retrieved_at="2026-06-12T10:30:00Z",
            truncated=False,
            content_tokens=42,
        )

        root = ElementTree.fromstring(prompt)
        self.assertEqual(root.tag, "url_grounded_answer")
        self.assertEqual(root.attrib["truncated"], "false")
        self.assertEqual(root.attrib["content_tokens"], "42")
        self.assertEqual(root.findtext("./page/title"), "\nExample Article\n")
        self.assertEqual(
            root.findtext("./page/url"), "\nhttps://example.com/article\n"
        )
        chunks = root.findall("./page/relevant_text/chunk")
        self.assertEqual(len(chunks), 2)
        self.assertIn("Cite the page URL", root.findtext("instructions") or "")

    def test_preserves_original_query_wording_after_outer_whitespace_trim(self) -> None:
        original = "  Does this page MENTION 'Async/Await' patterns?  "
        prompt = format_url_grounded_prompt(
            question=original,
            url="https://example.com/x",
            title="Title",
            ranked_chunks=[{"text": "chunk"}],
            today="2026-06-12",
        )

        root = ElementTree.fromstring(prompt)
        self.assertEqual(
            (root.findtext("question") or "").strip(),
            "Does this page MENTION 'Async/Await' patterns?",
        )

    def test_today_default_uses_utc_iso_date(self) -> None:
        prompt = format_url_grounded_prompt(
            question="q",
            url="https://example.com/x",
            title="Title",
            ranked_chunks=[{"text": "chunk"}],
        )

        root = ElementTree.fromstring(prompt)
        self.assertEqual(root.attrib["today"], datetime.now(UTC).date().isoformat())

    def test_no_chunks_uses_empty_relevant_text_element(self) -> None:
        prompt = format_url_grounded_prompt(
            question="q",
            url="https://example.com/x",
            title="Title",
            ranked_chunks=[],
            today="2026-06-12",
        )

        root = ElementTree.fromstring(prompt)
        relevant_text = root.find("./page/relevant_text")
        self.assertIsNotNone(relevant_text)
        assert relevant_text is not None
        self.assertEqual(list(relevant_text), [])


class FormatBrowseResultTests(unittest.TestCase):
    def _envelope(self, **stats_overrides) -> dict:
        stats = {
            "content_tokens": 10,
            "truncated": False,
            "session_id": "sess-1",
            "session_expires_in_seconds": 300,
            "actions_executed": 1,
        }
        stats.update(stats_overrides)
        return result_envelope(
            operation="browse",
            status="ok",
            query="pricing",
            retrieved_at="2026-06-12T10:30:00Z",
            sources=[{
                "id": "1",
                "url": "https://example.com/x",
                "title": "Title",
                "metadata": {},
                "chunks": [{"id": "1", "text": "Evidence.", "tokens": 2, "rank": 1, "scores": {}}],
                "links": [{"rank": 1, "url": "https://example.com/y", "text": "Next", "score": 0.9}],
            }],
            stats=stats,
        )

    def test_surfaces_session_id_and_expiry_as_root_attributes(self) -> None:
        rendered = format_browse_result(result=self._envelope(), today="2026-06-12")

        root = ElementTree.fromstring(rendered)
        self.assertEqual(root.tag, "browse_result")
        self.assertEqual(root.attrib["session_id"], "sess-1")
        self.assertEqual(root.attrib["session_expires_in_seconds"], "300")
        self.assertEqual(root.attrib["actions_executed"], "1")

    def test_renders_page_title_url_evidence_and_links(self) -> None:
        rendered = format_browse_result(result=self._envelope(), today="2026-06-12")

        root = ElementTree.fromstring(rendered)
        self.assertEqual(root.findtext("page/title", default="").strip(), "Title")
        self.assertEqual(root.findtext("page/url", default="").strip(), "https://example.com/x")
        self.assertIn("Evidence.", rendered)
        self.assertEqual(
            root.findtext("page/related_links/link/url", default="").strip(),
            "https://example.com/y",
        )

    def test_missing_session_id_renders_empty_attribute(self) -> None:
        rendered = format_browse_result(
            result=self._envelope(session_id=None), today="2026-06-12"
        )

        root = ElementTree.fromstring(rendered)
        self.assertEqual(root.attrib["session_id"], "")


if __name__ == "__main__":
    unittest.main()
