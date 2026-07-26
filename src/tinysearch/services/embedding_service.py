from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


DEFAULT_EMBEDDING_BACKEND = "onnx"
DEFAULT_EMBEDDING_OPENAI_ENV_FILE = ".env"
DEFAULT_EMBEDDING_MODEL = "fast"
FAST_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Backwards-compatible name for older imports/traces.
FIXED_LOCAL_EMBEDDING_MODEL = FAST_EMBEDDING_MODEL

SUPPORTED_EMBEDDING_BACKENDS = (
    "onnx",
    "openai_compatible",
)
LEGACY_ONNX_BACKEND_ALIASES = ("default", "local", "sentence_transformers")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EMBED_LOCK = threading.Lock()


@dataclass(frozen=True)
class LocalEmbeddingModelSpec:
    requested_model: str
    repo_id: str
    local_dir: Path
    onnx_paths: tuple[str, ...]
    pooling: str
    normalize: bool
    max_length: int
    allow_patterns: tuple[str, ...]
    is_preset: bool


_COMMON_ONNX_ALLOW_PATTERNS = (
    "model.onnx",
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer.model",
    "config.json",
)

_PRESET_MODELS: dict[str, dict[str, Any]] = {
    "fast": {
        "repo_id": "onnx-models/all-MiniLM-L6-v2-onnx",
        "local_dir": "all-minilm-l6-v2-onnx",
        "onnx_paths": ("model.onnx",),
        "pooling": "auto",
        "normalize": False,
        "max_length": 256,
        "allow_patterns": (
            "model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ),
    },
    "balanced": {
        "repo_id": "BAAI/bge-small-en-v1.5",
        "local_dir": "bge-small-en-v1.5-onnx",
        "onnx_paths": ("onnx/model.onnx", "model.onnx"),
        "pooling": "cls",
        "normalize": True,
        "max_length": 512,
        "allow_patterns": _COMMON_ONNX_ALLOW_PATTERNS,
    },
    "quality": {
        "repo_id": "BAAI/bge-base-en-v1.5",
        "local_dir": "bge-base-en-v1.5-onnx",
        "onnx_paths": ("onnx/model.onnx", "model.onnx"),
        "pooling": "cls",
        "normalize": True,
        "max_length": 512,
        "allow_patterns": _COMMON_ONNX_ALLOW_PATTERNS,
    },
}


def _models_dir() -> Path:
    raw_override = os.environ.get("TINYSEARCH_MODELS_DIR", "").strip()
    if raw_override:
        return Path(raw_override).expanduser().resolve()
    return (_PROJECT_ROOT / "models").resolve()


def _safe_model_dir_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip()).strip("-._")
    return f"{slug.lower() or 'custom-embedding-model'}-onnx"


def resolve_local_embedding_model_spec(
    embedding_model: str | None = None,
) -> LocalEmbeddingModelSpec:
    requested = (embedding_model or DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
    key = requested.lower()
    preset = _PRESET_MODELS.get(key)
    raw_override = os.environ.get("TINYSEARCH_ONNX_MODEL_DIR", "").strip()

    if preset is not None:
        local_dir = (
            Path(raw_override).expanduser().resolve()
            if raw_override
            else (_models_dir() / str(preset["local_dir"])).resolve()
        )
        return LocalEmbeddingModelSpec(
            requested_model=requested,
            repo_id=str(preset["repo_id"]),
            local_dir=local_dir,
            onnx_paths=tuple(preset["onnx_paths"]),
            pooling=str(preset["pooling"]),
            normalize=bool(preset["normalize"]),
            max_length=int(preset["max_length"]),
            allow_patterns=tuple(preset["allow_patterns"]),
            is_preset=True,
        )

    local_dir = (
        Path(raw_override).expanduser().resolve()
        if raw_override
        else (_models_dir() / _safe_model_dir_name(requested)).resolve()
    )
    return LocalEmbeddingModelSpec(
        requested_model=requested,
        repo_id=requested,
        local_dir=local_dir,
        onnx_paths=("model.onnx", "onnx/model.onnx"),
        pooling="auto",
        normalize=False,
        max_length=512,
        allow_patterns=_COMMON_ONNX_ALLOW_PATTERNS,
        is_preset=False,
    )


def _onnx_bundle_dir(embedding_model: str | None = None) -> Path:
    return resolve_local_embedding_model_spec(embedding_model).local_dir


def _find_onnx_model_path(spec: LocalEmbeddingModelSpec) -> Path | None:
    for rel in spec.onnx_paths:
        path = spec.local_dir / rel
        if path.is_file():
            return path
    for path in sorted(spec.local_dir.rglob("*.onnx")):
        return path
    return None


def _tokenizer_ready(bundle_dir: Path) -> bool:
    return (bundle_dir / "tokenizer.json").is_file()


def _onnx_bundle_ready(embedding_model: str | None = None) -> bool:
    spec = resolve_local_embedding_model_spec(embedding_model)
    return _find_onnx_model_path(spec) is not None and _tokenizer_ready(spec.local_dir)


def onnx_backend_will_use_onnx_bundle(
    embedding_model: str | None = None,
) -> bool:
    """True when the configured local embedding model has a usable ONNX bundle."""
    return _onnx_bundle_ready(embedding_model)


@dataclass(frozen=True)
class _LoadedOnnxBundle:
    session: Any
    tokenizer: Any
    spec: LocalEmbeddingModelSpec
    model_path: Path


@lru_cache(maxsize=8)
def _load_onnx_runtime_bundle_cached(
    embedding_model: str | None,
) -> _LoadedOnnxBundle:
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "ONNX embedding bundles require `onnxruntime` and `tokenizers`. "
            "Install with: pip install onnxruntime tokenizers"
        ) from exc

    spec = resolve_local_embedding_model_spec(embedding_model)
    model_path = _find_onnx_model_path(spec)
    if model_path is None or not _tokenizer_ready(spec.local_dir):
        raise RuntimeError(
            f"ONNX embedding bundle for {spec.requested_model!r} is incomplete under "
            f"{spec.local_dir}; start the server to download it or run ensure_onnx_bundle_sync()."
        )
    tokenizer = Tokenizer.from_file(str(spec.local_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=spec.max_length)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return _LoadedOnnxBundle(
        session=session,
        tokenizer=tokenizer,
        spec=spec,
        model_path=model_path,
    )


def _load_onnx_runtime_bundle(
    embedding_model: str | None = None,
) -> tuple[Any, Any]:
    loaded = _load_onnx_runtime_bundle_cached(embedding_model)
    return loaded.session, loaded.tokenizer


def clear_onnx_runtime_cache() -> None:
    """Drop cached ONNX sessions/tokenizers after replacing files under model dirs."""
    _load_onnx_runtime_bundle_cached.cache_clear()


def _normalize_rows(value: Any) -> Any:
    import numpy as np

    rows = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return rows / norms


def _pick_token_output(outputs: list[Any]) -> Any | None:
    for output in outputs:
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 3:
            return output
    for output in outputs:
        if hasattr(output, "ndim") and output.ndim == 3:
            return output
    return None


def _pick_pooled_output(outputs: list[Any]) -> Any | None:
    for output in outputs:
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 2:
            return output
    for output in outputs:
        if hasattr(output, "ndim") and output.ndim == 2:
            return output
    return None


def _pool_onnx_outputs(outputs: list[Any], spec: LocalEmbeddingModelSpec) -> Any:
    if spec.pooling == "cls":
        token_output = _pick_token_output(outputs)
        if token_output is None:
            raise RuntimeError(
                f"ONNX model {spec.repo_id!r} does not expose a BERT-like token output "
                "required for CLS pooling"
            )
        pooled = token_output[:, 0]
    else:
        pooled = _pick_pooled_output(outputs)
        if pooled is None:
            token_output = _pick_token_output(outputs)
            if token_output is None:
                raise RuntimeError(
                    f"ONNX model {spec.repo_id!r} has unsupported outputs; expected a "
                    "pooled 2D embedding output or a BERT-like 3D token output"
                )
            pooled = token_output[:, 0]

    if spec.normalize:
        pooled = _normalize_rows(pooled)
    return pooled


def _embed_onnx_sync(
    inputs: list[str],
    *,
    embedding_model: str | None = None,
) -> list[list[float]]:
    import numpy as np

    if not inputs:
        return []

    t0 = time.perf_counter()
    loaded = _load_onnx_runtime_bundle_cached(embedding_model)
    session = loaded.session
    tokenizer = loaded.tokenizer
    spec = loaded.spec
    t_after_load = time.perf_counter()
    n_chars = sum(len(s) for s in inputs)
    batch_size = 32
    all_rows: list[list[float]] = []
    with _EMBED_LOCK:
        t_embed0 = time.perf_counter()
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            encoded = tokenizer.encode_batch(batch)
            max_len = max((len(item.ids) for item in encoded), default=0)
            input_ids = np.asarray(
                [
                    item.ids + [0] * (max_len - len(item.ids))
                    for item in encoded
                ],
                dtype=np.int64,
            )
            attention_mask = np.asarray(
                [
                    item.attention_mask + [0] * (max_len - len(item.attention_mask))
                    for item in encoded
                ],
                dtype=np.int64,
            )
            input_names = {i.name for i in session.get_inputs()}
            ort_inputs: dict[str, Any] = {}
            if "input_ids" in input_names:
                ort_inputs["input_ids"] = input_ids
            if "attention_mask" in input_names:
                ort_inputs["attention_mask"] = attention_mask
            if "token_type_ids" in input_names:
                ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

            output_names = [o.name for o in session.get_outputs()]
            if "sentence_embedding" in output_names:
                out = session.run(("sentence_embedding",), ort_inputs)[0]
                if spec.normalize:
                    out = _normalize_rows(out)
            else:
                out = _pool_onnx_outputs(session.run(None, ort_inputs), spec)
            all_rows.extend(_as_vectors(out))
        embed_s = time.perf_counter() - t_embed0
    total_s = time.perf_counter() - t0
    if _embed_timing_log_enabled():
        prep_s = t_embed0 - t0
        lock_wait_s = t_embed0 - t_after_load
        print(
            f"[embedding] backend=onnx_cpu n_inputs={len(inputs)} chars={n_chars} "
            f"embed_s={embed_s:.3f} prep_s={prep_s:.3f} lock_wait_s={lock_wait_s:.3f} "
            f"total_s={total_s:.3f} model={spec.requested_model!r} "
            f"repo={spec.repo_id!r} bundle={spec.local_dir}",
            file=sys.stderr,
            flush=True,
        )
    return all_rows


def _embed_timing_log_enabled() -> bool:
    v = os.environ.get("TINYSEARCH_LOG_EMBED_TIMING", "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _as_vectors(value: Any) -> list[list[float]]:
    if hasattr(value, "reshape") and hasattr(value, "ndim") and value.ndim == 1:
        value = value.reshape(1, -1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    if isinstance(value[0], (int, float)):
        return [[float(x) for x in value]]
    vectors = [list(vector) for vector in value]
    return [[float(item) for item in vector] for vector in vectors]


def normalize_embedding_backend(backend: str) -> str:
    key = (backend or DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if key in ("onnx", "default", "local", "sentence_transformers"):
        return "onnx"
    if key in ("openai_compatible", "openai"):
        return "openai_compatible"
    if key == "llama_cpp":
        raise ValueError(
            "embedding_backend 'llama_cpp' is no longer supported; "
            "use 'onnx' (local ONNX embeddings) or 'openai_compatible' "
            "(credentials in .env)"
        )
    return key


def _resolve_openai_env_path(openai_env_file: str | Path | None) -> Path:
    raw = (openai_env_file or DEFAULT_EMBEDDING_OPENAI_ENV_FILE).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _parse_openai_env_file(path: Path) -> tuple[str | None, str, str]:
    """Read base URL, API key, and embedding model name from a .env-style file."""
    if not path.is_file():
        raise RuntimeError(
            f"openai_compatible backend requires {path} with OPENAI_BASE_URL (optional), "
            "OPENAI_API_KEY, and OPENAI_EMBEDDING_MODEL (or EMBEDDING_MODEL)"
        )
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        name, _, rest = stripped.partition("=")
        key = name.strip().upper()
        val = rest.strip().strip('"').strip("'")
        if key:
            values[key] = val

    def pick(*keys: str) -> str | None:
        for k in keys:
            v = values.get(k.upper())
            if v:
                return v
        return None

    api_key = pick("OPENAI_API_KEY", "API_KEY")
    if not api_key:
        raise RuntimeError(
            f"{path} must set OPENAI_API_KEY (or API_KEY) for openai_compatible embeddings"
        )
    base_raw = pick("OPENAI_BASE_URL", "BASE_URL", "API_URL")
    base_url = base_raw.strip() if base_raw else None
    if base_url == "":
        base_url = None
    model = pick(
        "OPENAI_EMBEDDING_MODEL",
        "EMBEDDING_MODEL",
        "MODEL_NAME",
        "MODEL",
    )
    if not model:
        raise RuntimeError(
            f"{path} must set OPENAI_EMBEDDING_MODEL (or EMBEDDING_MODEL / MODEL_NAME) "
            "for openai_compatible embeddings"
        )
    return base_url, api_key, model


def resolve_embedding_tokenizer_name(
    *,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: str | None = None,
    openai_env_file: str | Path | None = None,
) -> str:
    """Return the tokenizer source that matches the configured embedding backend."""
    backend_key = normalize_embedding_backend(backend)
    if backend_key == "onnx":
        return str(resolve_local_embedding_model_spec(embedding_model).local_dir)
    if backend_key == "openai_compatible":
        _, _, model_name = _parse_openai_env_file(_resolve_openai_env_path(openai_env_file))
        return model_name
    raise ValueError(
        f"unknown embedding_backend {backend!r}; expected one of {SUPPORTED_EMBEDDING_BACKENDS} "
        "(aliases: default, sentence_transformers, local -> onnx; openai -> openai_compatible)"
    )


# ---------------------------------------------------------------------------
# local ONNX backend.
# ---------------------------------------------------------------------------


def _create_onnx_embedder(
    embedding_model: str | None = None,
) -> Callable[[list[str]], Any]:
    async def embedder(inputs: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(
            _embed_onnx_sync,
            list(inputs),
            embedding_model=embedding_model,
        )

    return embedder


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (reads URL / key / model from .env at repo root).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_openai_client(base_url: str | None, api_key: str) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "The openai_compatible backend requires the `openai` package. "
            "Install with: pip install openai"
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs)
    except Exception as exc:
        raise RuntimeError("failed to construct OpenAI-compatible client") from exc


def _embed_openai_compatible_sync(
    client: Any,
    model_name: str,
    inputs: list[str],
    *,
    base_url: str | None,
) -> list[list[float]]:
    if not inputs:
        return []

    t0 = time.perf_counter()
    n_chars = sum(len(s) for s in inputs)
    try:
        response = client.embeddings.create(model=model_name, input=inputs)
    except Exception as exc:
        raise RuntimeError(
            f"failed to generate embeddings with openai_compatible model {model_name!r} "
            f"(base_url={base_url!r})"
        ) from exc
    embed_s = time.perf_counter() - t0
    if _embed_timing_log_enabled():
        print(
            f"[embedding] backend=openai_compatible n_inputs={len(inputs)} chars={n_chars} "
            f"embed_s={embed_s:.3f} model={model_name!r} base_url={base_url!r}",
            file=sys.stderr,
            flush=True,
        )

    return [list(item.embedding) for item in response.data]


def _create_openai_compatible_embedder(env_path: Path) -> Callable[[list[str]], Any]:
    base_url, api_key, model_name = _parse_openai_env_file(env_path)
    client = _load_openai_client(base_url, api_key)

    async def embedder(inputs: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(
            _embed_openai_compatible_sync,
            client,
            model_name,
            list(inputs),
            base_url=base_url,
        )

    return embedder


# ---------------------------------------------------------------------------
# Public factory.
# ---------------------------------------------------------------------------


def create_embedder(
    *,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: str | None = None,
    openai_env_file: str | Path | None = None,
) -> Callable[[list[str]], Any]:
    backend_key = normalize_embedding_backend(backend)
    if backend_key == "onnx":
        return _create_onnx_embedder(embedding_model)
    if backend_key == "openai_compatible":
        return _create_openai_compatible_embedder(_resolve_openai_env_path(openai_env_file))
    raise ValueError(
        f"unknown embedding_backend {backend!r}; expected one of {SUPPORTED_EMBEDDING_BACKENDS} "
        "(aliases: default, sentence_transformers, local -> onnx; openai -> openai_compatible)"
    )
