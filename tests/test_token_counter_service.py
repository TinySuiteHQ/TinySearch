from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinysearch.services.token_counter_service import (
    _looks_like_tokenizer_path,
    _resolve_tokenizers_json,
    resolve_tokenizer,
)


class TokenizerPathGuardTests(unittest.TestCase):
    def test_plain_encoding_names_are_not_paths(self) -> None:
        self.assertFalse(_looks_like_tokenizer_path("o200k_base"))
        self.assertFalse(_looks_like_tokenizer_path("text-embedding-3-small"))

    def test_path_like_names_are_detected(self) -> None:
        self.assertTrue(_looks_like_tokenizer_path("/tmp/tokenizer.json"))
        self.assertTrue(_looks_like_tokenizer_path("./models/tokenizer.json"))
        self.assertTrue(_looks_like_tokenizer_path("models/tokenizer.json"))

    def test_non_path_name_does_not_probe_filesystem(self) -> None:
        with patch("tinysearch.services.token_counter_service.Path") as path_cls:
            self.assertIsNone(_resolve_tokenizers_json("o200k_base"))
        path_cls.assert_not_called()

    def test_path_like_name_loads_tokenizer_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tokenizer_path = Path(td) / "tokenizer.json"
            tokenizer_path.write_text(
                '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{},"merges":[]}}',
                encoding="utf-8",
            )
            adapter = _resolve_tokenizers_json(str(tokenizer_path))
            self.assertIsNotNone(adapter)
            self.assertEqual(adapter.encode("hi"), [])

    def test_resolve_tokenizer_still_uses_tiktoken_for_o200k_base(self) -> None:
        tokenizer = resolve_tokenizer("o200k_base")
        self.assertGreater(len(tokenizer.encode("hello")), 0)


if __name__ == "__main__":
    unittest.main()
