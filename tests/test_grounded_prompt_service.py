from __future__ import annotations

import unittest

from tinysearch.services.grounded_prompt_service import (
    FIELD_RULE,
    PROMPT_RULE,
    format_relevant_text,
    format_search_grounded_prompt,
    format_url_grounded_prompt,
)


def _result_block() -> dict:
    return {
        "title": "Example Article",
        "url": "https://example.com/a",
        "snippet": "Short preview text.",
        "ranked_chunks": [
            {"text": "First relevant chunk about installation."},
            {"text": "Second chunk with another fact."},
        ],
    }


class FormatRelevantTextTests(unittest.TestCase):
    def test_emits_one_block_per_non_empty_chunk_keeping_original_ordinals(self) -> None:
        out = format_relevant_text(
            [{"text": "alpha"}, {"text": ""}, {"text": "beta"}]
        )

        self.assertIn("----- RELEVANT CHUNK 1 -----\nalpha", out)
        self.assertIn("----- RELEVANT CHUNK 3 -----\nbeta", out)
        self.assertNotIn("CHUNK 2", out)

    def test_returns_empty_string_when_no_chunks(self) -> None:
        self.assertEqual(format_relevant_text([]), "")


class FormatSearchGroundedPromptTests(unittest.TestCase):
    def test_preserves_existing_markers_and_format(self) -> None:
        prompt = format_search_grounded_prompt(
            question="What does it say about installation?",
            results=[_result_block()],
            today="2026-06-12",
        )

        self.assertIn("SEARCH-GROUNDED ANSWER PROMPT", prompt)
        self.assertIn("CRITICAL INSTRUCTIONS", prompt)
        self.assertIn(
            "You are answering the QUESTION using only the text under RESULTS.",
            prompt,
        )
        self.assertIn("RESULT 1", prompt)
        self.assertIn("TITLE 1\n======\nExample Article", prompt)
        self.assertIn("URL 1\n======\nhttps://example.com/a", prompt)
        self.assertIn(
            "SEARCH PREVIEW 1\n======\nShort preview text.",
            prompt,
        )
        self.assertIn("RELEVANT TEXT 1\n======", prompt)
        self.assertIn("----- RELEVANT CHUNK 1 -----", prompt)
        self.assertIn("First relevant chunk about installation.", prompt)
        self.assertIn("What does it say about installation?", prompt)
        self.assertEqual(prompt.count("\nQUESTION\n"), 2)
        self.assertEqual(prompt.count("\nTODAY\n"), 2)

    def test_empty_results_still_produces_question_section(self) -> None:
        prompt = format_search_grounded_prompt(
            question="zero hits",
            results=[],
            today="2026-06-12",
        )

        self.assertIn("RESULTS", prompt)
        self.assertNotIn("RESULT 1", prompt)
        self.assertIn("zero hits", prompt)


class FormatUrlGroundedPromptTests(unittest.TestCase):
    def test_contains_required_sections_and_url_specific_rules(self) -> None:
        prompt = format_url_grounded_prompt(
            question="What does this page say about installation?",
            url="https://example.com/article",
            title="Example Article",
            ranked_chunks=[{"text": "Run pip install."}, {"text": "Then run the CLI."}],
            today="2026-06-12",
        )

        self.assertIn("URL-GROUNDED ANSWER PROMPT", prompt)
        self.assertIn("CRITICAL INSTRUCTIONS", prompt)
        self.assertIn("PAGE", prompt)
        self.assertIn("TITLE\n======\nExample Article", prompt)
        self.assertIn("URL\n======\nhttps://example.com/article", prompt)
        self.assertIn("RELEVANT TEXT\n======", prompt)
        self.assertIn("----- RELEVANT CHUNK 1 -----\nRun pip install.", prompt)
        self.assertIn("----- RELEVANT CHUNK 2 -----\nThen run the CLI.", prompt)
        self.assertIn(
            "You are answering the QUESTION using only the text under PAGE.",
            prompt,
        )
        self.assertIn("Cite the page URL after each factual claim.", prompt)
        self.assertIn(
            "If the answer is not directly supported by the page, say the page is insufficient.",
            prompt,
        )
        self.assertIn(
            "Do not infer 'first', 'latest', or 'most recent' unless the page explicitly supports it.",
            prompt,
        )
        self.assertEqual(prompt.count("\nQUESTION\n"), 2)
        self.assertEqual(prompt.count("\nTODAY\n"), 2)

    def test_preserves_original_query_wording(self) -> None:
        original = "  Does this page MENTION 'Async/Await' patterns?  "
        prompt = format_url_grounded_prompt(
            question=original,
            url="https://example.com/x",
            title="Title",
            ranked_chunks=[{"text": "chunk"}],
            today="2026-06-12",
        )

        self.assertIn("Does this page MENTION 'Async/Await' patterns?", prompt)

    def test_today_default_uses_utc_iso_date(self) -> None:
        from datetime import UTC, datetime

        prompt = format_url_grounded_prompt(
            question="q",
            url="https://example.com/x",
            title="Title",
            ranked_chunks=[{"text": "chunk"}],
        )

        today = datetime.now(UTC).date().isoformat()
        self.assertIn(today, prompt)

    def test_no_relevant_text_section_when_chunks_empty(self) -> None:
        prompt = format_url_grounded_prompt(
            question="q",
            url="https://example.com/x",
            title="Title",
            ranked_chunks=[],
            today="2026-06-12",
        )

        self.assertNotIn("RELEVANT TEXT", prompt)
        self.assertIn("PAGE", prompt)


class ConstantsTests(unittest.TestCase):
    def test_prompt_rule_and_field_rule_are_exposed(self) -> None:
        self.assertEqual(PROMPT_RULE, "=" * 88)
        self.assertEqual(FIELD_RULE, "======")


if __name__ == "__main__":
    unittest.main()
