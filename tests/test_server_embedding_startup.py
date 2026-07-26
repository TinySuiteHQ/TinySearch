from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from tinysearch.core import _ensure_local_bundle_for_config
from tinysearch.servers.fastapi_server import (
    ResearchRequest,
    _tinysearch_version,
    research_endpoint,
)
from tinysearch.servers.mcp_server import _mcp_cors_origins, research as mcp_research
from tinysearch.services.tinysearch_config_service import (
    normalize_query,
)
from tinysearch.results import result_envelope


def _fn(coro):
    return getattr(coro, "fn", coro)


class EmbeddingStartupTests(unittest.IsolatedAsyncioTestCase):
    """Both adapters share one `tinysearch.core._ensure_local_bundle_for_config`."""

    async def test_ensures_selected_local_embedding_model(self) -> None:
        cfg = {"embedding_backend": "onnx", "embedding_model": "balanced"}

        with patch("tinysearch.services.onnx_bundle_service.ensure_onnx_bundle_sync") as ensure:
            await _ensure_local_bundle_for_config(cfg)

        ensure.assert_called_once_with("balanced")

    async def test_skips_openai_compatible_backend(self) -> None:
        cfg = {"embedding_backend": "openai_compatible", "embedding_model": "balanced"}

        with patch("tinysearch.services.onnx_bundle_service.ensure_onnx_bundle_sync") as ensure:
            await _ensure_local_bundle_for_config(cfg)

        ensure.assert_not_called()


class ResearchParityTests(unittest.IsolatedAsyncioTestCase):
    """Both adapters delegate to the same `tinysearch.core.research`."""

    async def test_research_uses_same_config_defaults_as_mcp(self) -> None:
        result = result_envelope(
            operation="research",
            status="ok",
            query="test query",
            sources=[],
        )
        core_research = AsyncMock(return_value=result)
        with patch("tinysearch.core.research", core_research):
            fastapi_response = await research_endpoint(
                ResearchRequest(query="  test query  ")
            )
            mcp_response = await _fn(mcp_research)("  test query  ")

        self.assertEqual(fastapi_response, mcp_response)
        self.assertIn("SEARCH-GROUNDED ANSWER PROMPT", fastapi_response["answer"])

    async def test_research_rejects_whitespace_only_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            await research_endpoint(ResearchRequest(query="   "))


class ServerRuntimeMetadataTests(unittest.TestCase):
    def test_version_comes_from_environment(self) -> None:
        with patch.dict("os.environ", {"TINYSEARCH_VERSION": "v0.2.0"}):
            self.assertEqual(_tinysearch_version(), "v0.2.0")

    def test_version_defaults_to_dev(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_tinysearch_version(), "dev")

    def test_query_normalization_is_shared(self) -> None:
        self.assertEqual(normalize_query("  hello  "), "hello")
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            normalize_query("  ")


class McpCorsConfigTests(unittest.TestCase):
    def test_cors_origins_default_to_wildcard(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_mcp_cors_origins(), ["*"])

    def test_cors_origins_parse_comma_separated_list(self) -> None:
        with patch.dict(
            "os.environ",
            {"MCP_CORS_ORIGINS": "http://localhost:8080, http://172.20.210.53:8080"},
        ):
            self.assertEqual(
                _mcp_cors_origins(),
                ["http://localhost:8080", "http://172.20.210.53:8080"],
            )


if __name__ == "__main__":
    unittest.main()
