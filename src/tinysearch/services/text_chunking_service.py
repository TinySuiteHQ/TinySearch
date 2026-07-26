from __future__ import annotations

from services.token_counter_service import resolve_tokenizer


def truncate_text_to_max_tokens(
    text: str,
    max_tokens: int | None,
    encoding_name: str | None = None,
    *,
    model: str | None = None,
) -> str:
    """Keep the start of ``text`` up to ``max_tokens`` (inclusive of tokenizer).

    ``max_tokens`` <= 0 or ``None`` means do not truncate.
    """

    if not text or max_tokens is None or max_tokens <= 0:
        return text
    encoding = resolve_tokenizer(encoding_name, model=model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def chunk_text(
    text: str,
    max_chunk_tokens: int = 500,
    overlap_tokens: int = 80,
    encoding_name: str | None = None,
    *,
    model: str | None = None,
) -> list[dict]:
    """
    Structure-aware text chunking.

    Markdown headings are preserved as chunk metadata when present, but callers
    can pass any plain text.
    """
    encoding = resolve_tokenizer(encoding_name, model=model)

    import re

    blocks = re.split(r"(?=\n#{1,6}\s)|\n{2,}", text.strip())

    chunks: list[dict] = []
    current_blocks: list[str] = []
    current_heading = ""

    def flush() -> None:
        nonlocal current_blocks, current_heading

        chunk_body = "\n\n".join(block.strip() for block in current_blocks if block.strip()).strip()
        if not chunk_body:
            current_blocks = []
            return

        chunks.append(
            {
                "chunk_id": len(chunks),
                "heading": current_heading,
                "text": chunk_body,
                "tokens": len(encoding.encode(chunk_body)),
            }
        )
        current_blocks = []

    def split_large_block(block: str) -> None:
        tokens = encoding.encode(block)
        step = max_chunk_tokens - overlap_tokens
        if step <= 0:
            step = max_chunk_tokens

        for start in range(0, len(tokens), step):
            end = min(start + max_chunk_tokens, len(tokens))
            piece = encoding.decode(tokens[start:end]).strip()
            if piece:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "heading": current_heading,
                        "text": piece,
                        "tokens": len(encoding.encode(piece)),
                    }
                )

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        heading = _parse_markdown_heading(block)
        if heading is not None:
            current_heading = heading

        block_tokens = len(encoding.encode(block))

        if block_tokens > max_chunk_tokens:
            flush()
            split_large_block(block)
            continue

        candidate = "\n\n".join(current_blocks + [block]).strip()
        candidate_tokens = len(encoding.encode(candidate))

        if candidate_tokens <= max_chunk_tokens:
            current_blocks.append(block)
        else:
            flush()
            current_blocks.append(block)

    flush()
    return chunks

chunk_markdown = chunk_text


def _parse_markdown_heading(block: str) -> str | None:
    if not block or block[0] != "#":
        return None
    marker_count = 0
    while marker_count < len(block) and block[marker_count] == "#":
        marker_count += 1
    if marker_count < 1 or marker_count > 6:
        return None
    if marker_count >= len(block) or not block[marker_count].isspace():
        return None
    heading = block[marker_count:].strip()
    return heading or None
