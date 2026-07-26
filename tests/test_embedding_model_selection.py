from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tinysearch.services import embedding_service
from tinysearch.services.embedding_service import (
    LocalEmbeddingModelSpec,
    _LoadedOnnxBundle,
    _embed_onnx_sync,
    _pool_onnx_outputs,
    normalize_embedding_backend,
    resolve_local_embedding_model_spec,
)
from tinysearch.services.onnx_bundle_service import ensure_onnx_bundle_sync
from tinysearch.services.tinysearch_config_service import load_tinysearch_config, tokenizer_name_for_config


class EmbeddingModelSelectionTests(unittest.TestCase):
    def test_default_config_uses_fast_embedding_model(self) -> None:
        cfg = load_tinysearch_config()

        self.assertEqual(cfg["embedding_model"], "fast")

    def test_builtin_presets_resolve_to_expected_repos_and_dirs(self) -> None:
        fast = resolve_local_embedding_model_spec("fast")
        balanced = resolve_local_embedding_model_spec("balanced")
        quality = resolve_local_embedding_model_spec("quality")

        self.assertEqual(fast.repo_id, "onnx-models/all-MiniLM-L6-v2-onnx")
        self.assertEqual(fast.local_dir.name, "all-minilm-l6-v2-onnx")
        self.assertEqual(balanced.repo_id, "BAAI/bge-small-en-v1.5")
        self.assertEqual(balanced.local_dir.name, "bge-small-en-v1.5-onnx")
        self.assertEqual(quality.repo_id, "BAAI/bge-base-en-v1.5")
        self.assertEqual(quality.local_dir.name, "bge-base-en-v1.5-onnx")

    def test_custom_hf_repo_resolves_to_deterministic_models_dir(self) -> None:
        spec = resolve_local_embedding_model_spec("some-org/some-onnx-embedding-repo")

        self.assertEqual(spec.repo_id, "some-org/some-onnx-embedding-repo")
        self.assertEqual(spec.local_dir.name, "some-org-some-onnx-embedding-repo-onnx")
        self.assertFalse(spec.is_preset)

    def test_models_dir_env_sets_model_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(
                os.environ,
                {"TINYSEARCH_MODELS_DIR": td, "TINYSEARCH_ONNX_MODEL_DIR": ""},
                clear=False,
            ):
                balanced = resolve_local_embedding_model_spec("balanced")
                custom = resolve_local_embedding_model_spec("some-org/custom-model")

        self.assertEqual(
            balanced.local_dir,
            (Path(td) / "bge-small-en-v1.5-onnx").resolve(),
        )
        self.assertEqual(
            custom.local_dir,
            (Path(td) / "some-org-custom-model-onnx").resolve(),
        )

    def test_exact_onnx_model_dir_env_overrides_models_root(self) -> None:
        with tempfile.TemporaryDirectory() as models_td:
            with tempfile.TemporaryDirectory() as bundle_td:
                with patch.dict(
                    os.environ,
                    {
                        "TINYSEARCH_MODELS_DIR": models_td,
                        "TINYSEARCH_ONNX_MODEL_DIR": bundle_td,
                    },
                    clear=False,
                ):
                    spec = resolve_local_embedding_model_spec("quality")

        self.assertEqual(spec.local_dir, Path(bundle_td).resolve())

    def test_config_path_env_loads_mounted_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "tinysearch_config.json"
            config_path.write_text(
                '{"embedding_backend": "onnx", "embedding_model": "balanced"}',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYSEARCH_CONFIG_PATH": str(config_path)}):
                cfg = load_tinysearch_config()

        self.assertEqual(cfg["embedding_model"], "balanced")
        self.assertEqual(cfg["embedding_backend"], "onnx")

    def test_config_path_env_loads_blocked_domains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "tinysearch_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "embedding_backend": "onnx",
                        "embedding_model": "balanced",
                        "blocked_domains": [
                            "example.com",
                            "",
                            "https://spammy-site.test/path",
                            None,
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TINYSEARCH_CONFIG_PATH": str(config_path)}):
                cfg = load_tinysearch_config()

        self.assertEqual(
            cfg["blocked_domains"],
            ["example.com", "spammy-site.test"],
        )

    def test_embedding_tokenizer_uses_selected_model_bundle_dir(self) -> None:
        cfg = {
            "encoding_name": "embedding",
            "embedding_backend": "onnx",
            "embedding_model": "balanced",
            "embedding_openai_env_file": ".env",
        }

        self.assertEqual(
            Path(tokenizer_name_for_config(cfg)).name,
            "bge-small-en-v1.5-onnx",
        )

    def test_explicit_encoding_name_bypasses_embedding_tokenizer(self) -> None:
        cfg = {
            "encoding_name": "o200k_base",
            "embedding_backend": "onnx",
            "embedding_model": "quality",
            "embedding_openai_env_file": ".env",
        }

        self.assertEqual(tokenizer_name_for_config(cfg), "o200k_base")

    def test_legacy_default_backend_aliases_to_onnx(self) -> None:
        self.assertEqual(normalize_embedding_backend("default"), "onnx")


class OnnxBundleDownloadTests(unittest.TestCase):
    def test_complete_bundle_is_not_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td)
            (bundle_dir / "model.onnx").write_bytes(b"onnx")
            (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {"TINYSEARCH_ONNX_MODEL_DIR": td}):
                with patch("tinysearch.services.onnx_bundle_service.snapshot_download") as download:
                    ensure_onnx_bundle_sync("fast")

            download.assert_not_called()

    def test_missing_bundle_downloads_to_selected_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            def fake_download(**kwargs):
                local_dir = Path(kwargs["local_dir"])
                (local_dir / "model.onnx").write_bytes(b"onnx")
                (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {"TINYSEARCH_ONNX_MODEL_DIR": td}):
                with patch(
                    "tinysearch.services.onnx_bundle_service.snapshot_download",
                    side_effect=fake_download,
                ) as download:
                    ensure_onnx_bundle_sync("fast")

            download.assert_called_once()
            self.assertTrue((Path(td) / "model.onnx").is_file())
            self.assertTrue((Path(td) / "tokenizer.json").is_file())

    def test_incomplete_download_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            def fake_download(**kwargs):
                local_dir = Path(kwargs["local_dir"])
                (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {"TINYSEARCH_ONNX_MODEL_DIR": td}):
                with patch(
                    "tinysearch.services.onnx_bundle_service.snapshot_download",
                    side_effect=fake_download,
                ):
                    with self.assertRaisesRegex(RuntimeError, "incomplete"):
                        ensure_onnx_bundle_sync("fast")


class OnnxInferenceTests(unittest.TestCase):
    def test_bge_style_cls_pooling_normalizes_vectors(self) -> None:
        spec = LocalEmbeddingModelSpec(
            requested_model="balanced",
            repo_id="BAAI/bge-small-en-v1.5",
            local_dir=Path("/tmp/model"),
            onnx_paths=("onnx/model.onnx",),
            pooling="cls",
            normalize=True,
            max_length=512,
            allow_patterns=(),
            is_preset=True,
        )
        token_output = np.array(
            [
                [[3.0, 4.0], [100.0, 100.0]],
                [[0.0, 2.0], [100.0, 100.0]],
            ],
            dtype=np.float32,
        )

        pooled = _pool_onnx_outputs([token_output], spec)

        np.testing.assert_allclose(
            pooled,
            np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
        )

    def test_unsupported_custom_output_raises_clear_error(self) -> None:
        spec = LocalEmbeddingModelSpec(
            requested_model="custom/model",
            repo_id="custom/model",
            local_dir=Path("/tmp/model"),
            onnx_paths=("model.onnx",),
            pooling="auto",
            normalize=False,
            max_length=512,
            allow_patterns=(),
            is_preset=False,
        )

        with self.assertRaisesRegex(RuntimeError, "unsupported outputs"):
            _pool_onnx_outputs([np.array([1.0, 2.0], dtype=np.float32)], spec)

    def test_sentence_embedding_output_is_used_directly(self) -> None:
        class FakeIo:
            def __init__(self, name: str) -> None:
                self.name = name

        class FakeEncoded:
            ids = [1, 2]
            attention_mask = [1, 1]

        class FakeTokenizer:
            def encode_batch(self, _batch):
                return [FakeEncoded()]

        class FakeSession:
            def get_inputs(self):
                return [FakeIo("input_ids"), FakeIo("attention_mask")]

            def get_outputs(self):
                return [FakeIo("sentence_embedding")]

            def run(self, names, _inputs):
                self.names = names
                return [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]

        spec = LocalEmbeddingModelSpec(
            requested_model="fast",
            repo_id="onnx-models/all-MiniLM-L6-v2-onnx",
            local_dir=Path("/tmp/model"),
            onnx_paths=("model.onnx",),
            pooling="auto",
            normalize=False,
            max_length=256,
            allow_patterns=(),
            is_preset=True,
        )
        loaded = _LoadedOnnxBundle(
            session=FakeSession(),
            tokenizer=FakeTokenizer(),
            spec=spec,
            model_path=Path("/tmp/model/model.onnx"),
        )

        with patch.object(
            embedding_service,
            "_load_onnx_runtime_bundle_cached",
            return_value=loaded,
        ):
            vectors = _embed_onnx_sync(["hello"], embedding_model="fast")

        self.assertEqual(vectors, [[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
