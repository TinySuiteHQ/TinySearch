from __future__ import annotations

import unittest

from tinysearch.services.text_chunking_service import (
    _parse_aria_heading,
    _parse_markdown_heading,
    chunk_text,
    truncate_text_to_max_tokens,
)


class TruncateTextToMaxTokensTests(unittest.TestCase):
    def test_zero_or_negative_means_no_truncate(self) -> None:
        text = "hello " * 100
        self.assertIs(truncate_text_to_max_tokens(text, 0, "o200k_base"), text)
        self.assertIs(truncate_text_to_max_tokens(text, -1, "o200k_base"), text)

    def test_none_means_no_truncate(self) -> None:
        text = "abc"
        self.assertIs(truncate_text_to_max_tokens(text, None, "o200k_base"), text)

    def test_shortens_to_token_budget(self) -> None:
        text = "word " * 400
        out = truncate_text_to_max_tokens(text, 12, "o200k_base")
        self.assertLess(len(out), len(text))
        self.assertTrue(out.strip())

    def test_parse_markdown_heading(self) -> None:
        self.assertEqual(_parse_markdown_heading("# Intro"), "Intro")
        self.assertEqual(_parse_markdown_heading("###### Deep"), "Deep")
        self.assertIsNone(_parse_markdown_heading("not a heading"))
        self.assertIsNone(_parse_markdown_heading("####### Too many"))

    def test_parse_aria_heading(self) -> None:
        self.assertEqual(
            _parse_aria_heading('- heading "Installation" [level=2]'),
            "Installation",
        )
        self.assertEqual(
            _parse_aria_heading('- heading "API Reference" [level=2] [ref=e42]'),
            "API Reference",
        )
        self.assertIsNone(_parse_aria_heading("- paragraph: not a heading"))

    def test_chunk_text_splits_aria_sections_without_blank_lines(self) -> None:
        chunks = chunk_text(
            '- heading "Intro" [level=1]\n'
            '- paragraph: First section text.\n'
            '- paragraph: Still first section.\n'
            '- heading "Install" [level=2]\n'
            '- paragraph: pip install tinysuite-search',
            max_chunk_tokens=500,
            encoding_name="o200k_base",
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["heading"], "Intro")
        self.assertEqual(chunks[1]["heading"], "Install")
        self.assertIn("First section text", chunks[0]["text"])
        self.assertIn("pip install tinysuite-search", chunks[1]["text"])

    def test_oversized_aria_section_is_windowed_with_heading_preserved(self) -> None:
        tail = "TAIL_SENTINEL"
        text = (
            '- heading "Long section" [level=2]\n'
            + '- paragraph: evidence ' * 200
            + tail
        )
        chunks = chunk_text(
            text,
            max_chunk_tokens=20,
            overlap_tokens=5,
            encoding_name="o200k_base",
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["tokens"] <= 20 for chunk in chunks))
        self.assertTrue(all(chunk["heading"] == "Long section" for chunk in chunks))
        self.assertIn(tail, chunks[-1]["text"])

    def test_unheaded_oversized_text_still_uses_token_windows(self) -> None:
        text = "plain evidence " * 100
        chunks = chunk_text(
            text,
            max_chunk_tokens=20,
            overlap_tokens=5,
            encoding_name="o200k_base",
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["tokens"] <= 20 for chunk in chunks))

    def test_chunk_text_preserves_heading_metadata(self) -> None:
        chunks = chunk_text(
            "# Only Section\n\nBody text.",
            max_chunk_tokens=500,
            encoding_name="o200k_base",
        )
        self.assertEqual(chunks[0]["heading"], "Only Section")
        self.assertIn("Body text.", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
