from __future__ import annotations

import unittest

from tinysearch.services.text_chunking_service import _parse_markdown_heading, chunk_text, truncate_text_to_max_tokens


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
