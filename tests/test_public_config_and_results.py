from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinysearch import TinySearchConfig, to_prompt
from tinysearch.config import resolve_config
from tinysearch.results import result_envelope
from tinysearch.services.tinysearch_config_service import load_tinysearch_config


class PublicConfigTests(unittest.TestCase):
    def test_defaults_are_zero_infrastructure_and_json_serializable(self) -> None:
        config = TinySearchConfig()
        self.assertEqual(config["search_backend"], "ddgs")
        self.assertEqual(config.search_backend, "ddgs")
        self.assertEqual(config.browser_cdp_url, "")
        json.dumps(config.to_dict())

    def test_explicit_file_then_call_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(
                json.dumps({"search_top_k": 20, "search_backend": "searxng"}),
                encoding="utf-8",
            )
            config = resolve_config({"search_top_k": 3}, path=path)
        self.assertEqual(config["search_backend"], "searxng")
        self.assertEqual(config["search_top_k"], 3)

    def test_core_config_ignores_environment_and_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "TINYSEARCH_CONFIG_PATH": "/does/not/matter.json",
                "SEARXNG_URL": "http://example.test/search",
            },
        ), patch("os.getcwd", return_value=temp_dir):
            config = TinySearchConfig()
        self.assertEqual(config["search_backend"], "ddgs")
        self.assertNotEqual(
            config["search_backend_url"],
            "http://example.test/search",
        )

    def test_server_loader_applies_environment_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TINYSEARCH_CONFIG_PATH": "/does/not/exist.json",
                "TINYSEARCH_SEARCH_BACKEND": "searxng",
                "SEARXNG_URL": "http://example.test/search",
            },
            clear=True,
        ):
            config = load_tinysearch_config()
        self.assertEqual(config["search_backend"], "searxng")
        self.assertEqual(config["search_backend_url"], "http://example.test/search")

    def test_comment_fields_are_ignored(self) -> None:
        config = TinySearchConfig.from_mapping(
            {
                "_comment_embedding_model": "Choose a local model.",
                "embedding_model": "balanced",
            }
        )
        self.assertEqual(config.embedding_model, "balanced")
        self.assertNotIn("_comment_embedding_model", config)

    def test_embedding_backend_is_normalized_and_validated(self) -> None:
        self.assertEqual(
            TinySearchConfig(embedding_backend="openai").embedding_backend,
            "openai_compatible",
        )
        with self.assertRaisesRegex(ValueError, "embedding_backend"):
            TinySearchConfig(embedding_backend="unknown")

    def test_server_loader_applies_embedding_environment_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TINYSEARCH_CONFIG_PATH": "/does/not/exist.json",
                "TINYSEARCH_EMBEDDING_BACKEND": "onnx",
                "TINYSEARCH_EMBEDDING_MODEL": "quality",
            },
            clear=True,
        ):
            config = load_tinysearch_config()
        self.assertEqual(config["embedding_backend"], "onnx")
        self.assertEqual(config["embedding_model"], "quality")

    def test_browser_cdp_url_is_normalized_and_validated(self) -> None:
        config = TinySearchConfig(browser_cdp_url="  http://browser:9222  ")
        self.assertEqual(config.browser_cdp_url, "http://browser:9222")

        for invalid_url in ("browser:9222", "ftp://browser:9222", "ws:///devtools"):
            with self.subTest(invalid_url=invalid_url), self.assertRaisesRegex(
                ValueError, "browser_cdp_url"
            ):
                TinySearchConfig(browser_cdp_url=invalid_url)

    def test_server_loader_applies_browser_cdp_environment_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TINYSEARCH_CONFIG_PATH": "/does/not/exist.json",
                "TINYSEARCH_BROWSER_CDP_URL": "ws://browser:9222/devtools/browser/id",
            },
            clear=True,
        ):
            config = load_tinysearch_config()
        self.assertEqual(
            config["browser_cdp_url"],
            "ws://browser:9222/devtools/browser/id",
        )


class PublicResultTests(unittest.TestCase):
    def test_prompt_renderer_is_deterministic(self) -> None:
        result = result_envelope(
            operation="scrape",
            status="ok",
            query="What happened?",
            retrieved_at="2026-01-01T00:00:00Z",
            sources=[{
                "id": "1",
                "title": "Source",
                "url": "https://example.com",
                "snippet": "Preview",
                "chunks": [{
                    "id": "1:1",
                    "text": "Evidence",
                    "tokens": 1,
                    "rank": 1,
                    "scores": {"rrf": 1.0, "dense": 1.0, "bm25": 1.0},
                }],
            }],
        )
        first = to_prompt(result, today="2026-07-26")
        second = to_prompt(result, today="2026-07-26")
        self.assertEqual(first, second)
        self.assertIn("Evidence", first)

    def test_prompt_renderer_rejects_unknown_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            to_prompt({"schema_version": "2", "operation": "scrape"})


if __name__ == "__main__":
    unittest.main()
