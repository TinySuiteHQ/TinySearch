from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_ENCODING_NAME = "o200k_base"
EMBEDDING_TOKENIZER_NAME = "embedding"


class HuggingFaceTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class CharacterTokenizerAdapter:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class TiktokenAdapter:
    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


class TokenizersAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        no_trunc = getattr(self._tokenizer, "no_truncation", None)
        if callable(no_trunc):
            no_trunc()
        no_pad = getattr(self._tokenizer, "no_padding", None)
        if callable(no_pad):
            no_pad()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


def _tiktoken_module():
    try:
        import tiktoken
    except Exception:
        return None
    return tiktoken


def _get_tiktoken_encoding(name: str):
    tiktoken = _tiktoken_module()
    if tiktoken is None:
        return None
    try:
        return TiktokenAdapter(tiktoken.get_encoding(name))
    except Exception:
        return None


def _get_tiktoken_encoding_for_model(name: str):
    tiktoken = _tiktoken_module()
    if tiktoken is None:
        return None
    try:
        return TiktokenAdapter(tiktoken.encoding_for_model(name))
    except Exception:
        return None


def _resolve_huggingface_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return None
    return HuggingFaceTokenizerAdapter(tokenizer)


def _looks_like_tokenizer_path(name: str) -> bool:
    if name.startswith(("/", "./", "../")):
        return True
    if "/" in name or "\\" in name:
        return True
    return name.lower().endswith(".json")


def _resolve_tokenizers_json(name: str):
    if not _looks_like_tokenizer_path(name):
        return None
    try:
        from tokenizers import Tokenizer
    except Exception:
        return None

    path = Path(name).expanduser()
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        return None
    try:
        return TokenizersAdapter(Tokenizer.from_file(str(path)))
    except Exception:
        return None


def normalize_tokenizer_name(tokenizer_name: str | None) -> str | None:
    if tokenizer_name is None:
        return None
    name = str(tokenizer_name).strip()
    if not name or name.lower() == EMBEDDING_TOKENIZER_NAME:
        return None
    return name


@lru_cache(maxsize=64)
def resolve_tokenizer(
    tokenizer_name: str | None = None,
    *,
    model: str | None = None,
):
    """
    Resolve a tokenizer for chunking and token counting.

    Default config uses an explicit tiktoken encoding (e.g. ``o200k_base``). That keeps
    chunking simple; token counts may not match the ONNX embedder's tokenizer unless
    ``encoding_name`` is set to ``embedding`` (bundle ``tokenizer.json``).
    """
    tokenizer_name = normalize_tokenizer_name(tokenizer_name)
    if tokenizer_name:
        encoding = _get_tiktoken_encoding(tokenizer_name)
        if encoding is not None:
            return encoding
        tokenizers_json = _resolve_tokenizers_json(tokenizer_name)
        if tokenizers_json is not None:
            return tokenizers_json
        hf_tokenizer = _resolve_huggingface_tokenizer(tokenizer_name)
        if hf_tokenizer is not None:
            return hf_tokenizer

    candidates = [tokenizer_name, model]
    for candidate in candidates:
        if not candidate:
            continue
        tokenizers_json = _resolve_tokenizers_json(candidate)
        if tokenizers_json is not None:
            return tokenizers_json
        hf_tokenizer = _resolve_huggingface_tokenizer(candidate)
        if hf_tokenizer is not None:
            return hf_tokenizer
        encoding = _get_tiktoken_encoding_for_model(candidate)
        if encoding is not None:
            return encoding
        encoding = _get_tiktoken_encoding(candidate)
        if encoding is not None:
            return encoding
    fallback = _get_tiktoken_encoding(DEFAULT_ENCODING_NAME)
    if fallback is not None:
        return fallback
    return CharacterTokenizerAdapter()


def token_count(
    text: str,
    encoding_name: str | None = DEFAULT_ENCODING_NAME,
    *,
    model: str | None = None,
) -> int:
    encoding = resolve_tokenizer(encoding_name, model=model)
    return len(encoding.encode(text))


def encode_tokens(
    text: str,
    encoding_name: str | None = DEFAULT_ENCODING_NAME,
    *,
    model: str | None = None,
) -> list[int]:
    return resolve_tokenizer(encoding_name, model=model).encode(text)


def decode_tokens(
    tokens: list[int],
    encoding_name: str | None = DEFAULT_ENCODING_NAME,
    *,
    model: str | None = None,
) -> str:
    return resolve_tokenizer(encoding_name, model=model).decode(tokens)
