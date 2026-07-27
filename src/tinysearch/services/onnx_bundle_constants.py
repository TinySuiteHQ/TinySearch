"""HF repos used for MiniLM ONNX embeddings.

PyTorch export uses ``MINILM_SENTENCE_TRANSFORMERS_REPO_ID``. Runtime download uses a
pre-built ONNX repo whose graph exposes ``sentence_embedding`` (mean-pooled). The ONNX
file under ``sentence-transformers/.../onnx/`` on Hugging Face is a different graph
(sequence ``last_hidden_state``), so it is not used for TinySearch.
"""

from __future__ import annotations

MINILM_SENTENCE_TRANSFORMERS_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Pre-built bundle (tokenizer + ONNX). License: Apache-2.0 per Hugging Face model card.
MINILM_ONNX_BUNDLE_REPO_ID = "onnx-models/all-MiniLM-L6-v2-onnx"

MINILM_ONNX_BUNDLE_ALLOW_PATTERNS: tuple[str, ...] = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
