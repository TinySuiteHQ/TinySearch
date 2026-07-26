"""Shared, testable prompt builders for grounded answer prompts.

Both `/research` (search-grounded, multi-source) and `/scrape` (URL-grounded,
single-source) construct answer prompts in the same family of formats. Keeping
the builders here avoids drift between adapters and lets us cover them with
focused tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any


PROMPT_RULE = "=" * 88
FIELD_RULE = "======"


def format_relevant_text(chunks: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for ordinal, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        blocks.extend(
            [
                f"----- RELEVANT CHUNK {ordinal} -----",
                text,
            ]
        )
    return "\n".join(blocks).strip()


def _today_text(today: str | None) -> str:
    return today or datetime.now(UTC).date().isoformat()


def format_search_grounded_prompt(
    *,
    question: str,
    results: Sequence[dict[str, Any]],
    today: str | None = None,
) -> str:
    clean_question = question.strip()
    today_text = _today_text(today)
    lines = [
        PROMPT_RULE,
        "SEARCH-GROUNDED ANSWER PROMPT",
        PROMPT_RULE,
        "",
        "QUESTION",
        PROMPT_RULE,
        clean_question,
        PROMPT_RULE,
        "",
        "TODAY",
        PROMPT_RULE,
        today_text,
        PROMPT_RULE,
        "",
        "CRITICAL INSTRUCTIONS",
        PROMPT_RULE,
        "You are answering the QUESTION using only the text under RESULTS.",
        "First resolve any relative date in the QUESTION using TODAY.",
        f"TODAY is {today_text!r}.",
        f"For example, 'last year' means calendar year {int(today_text.split('-')[0]) - 1}.",
        "Use only facts directly supported by RESULTS.",
        "Do not use your own knowledge.",
        "Do not add extra historical claims unless directly supported by RESULTS.",
        "Do not infer 'first ever', 'most recent', 'record', or franchise history unless RESULTS explicitly support it.",
        "If RESULTS contain conflicting information, prefer the result that directly matches the resolved date and question.",
        "If the conflict cannot be resolved, say the results conflict.",
        "Cite the source URL after each factual claim.",
        "If the answer is not directly supported by RESULTS, say the results are not enough.",
        PROMPT_RULE,
        "",
        PROMPT_RULE,
        "RESULTS",
        PROMPT_RULE,
        "",
    ]

    for ordinal, result in enumerate(results, start=1):
        relevant_text = format_relevant_text(result.get("ranked_chunks") or [])
        lines.extend(
            [
                PROMPT_RULE,
                f"RESULT {ordinal}",
                PROMPT_RULE,
                f"TITLE {ordinal}",
                FIELD_RULE,
                str(result["title"]).strip(),
                FIELD_RULE,
                f"URL {ordinal}",
                FIELD_RULE,
                str(result["url"]).strip(),
                FIELD_RULE,
                f"SEARCH PREVIEW {ordinal}",
                FIELD_RULE,
                str(result.get("snippet") or "").strip(),
                FIELD_RULE,
            ]
        )
        if relevant_text:
            lines.extend(
                [
                    f"RELEVANT TEXT {ordinal}",
                    FIELD_RULE,
                    relevant_text,
                    FIELD_RULE,
                ]
            )
        lines.append("")

    lines.extend(
        [
            PROMPT_RULE,
            "QUESTION",
            PROMPT_RULE,
            clean_question,
            PROMPT_RULE,
            "",
            "TODAY",
            PROMPT_RULE,
            today_text,
            PROMPT_RULE,
            "",
            PROMPT_RULE,
            "SEARCH-GROUNDED ANSWER PROMPT",
            PROMPT_RULE,
        ]
    )
    return "\n".join(lines).strip()


def format_url_grounded_prompt(
    *,
    question: str,
    url: str,
    title: str,
    ranked_chunks: Sequence[dict[str, Any]],
    today: str | None = None,
) -> str:
    clean_question = question.strip()
    today_text = _today_text(today)
    lines = [
        PROMPT_RULE,
        "URL-GROUNDED ANSWER PROMPT",
        PROMPT_RULE,
        "",
        "QUESTION",
        PROMPT_RULE,
        clean_question,
        PROMPT_RULE,
        "",
        "TODAY",
        PROMPT_RULE,
        today_text,
        PROMPT_RULE,
        "",
        "CRITICAL INSTRUCTIONS",
        PROMPT_RULE,
        "You are answering the QUESTION using only the text under PAGE.",
        "First resolve any relative date in the QUESTION using TODAY.",
        f"TODAY is {today_text!r}.",
        f"For example, 'last year' means calendar year {int(today_text.split('-')[0]) - 1}.",
        "Use only facts directly supported by the page evidence.",
        "Do not use your own knowledge.",
        "Do not add extra historical claims unless directly supported by the page.",
        "Do not infer 'first', 'latest', or 'most recent' unless the page explicitly supports it.",
        "Cite the page URL after each factual claim.",
        "If the answer is not directly supported by the page, say the page is insufficient.",
        PROMPT_RULE,
        "",
        PROMPT_RULE,
        "PAGE",
        PROMPT_RULE,
        "",
        "TITLE",
        FIELD_RULE,
        str(title).strip(),
        FIELD_RULE,
        "URL",
        FIELD_RULE,
        str(url).strip(),
        FIELD_RULE,
    ]

    relevant_text = format_relevant_text(ranked_chunks)
    if relevant_text:
        lines.extend(
            [
                "RELEVANT TEXT",
                FIELD_RULE,
                relevant_text,
                FIELD_RULE,
            ]
        )
    lines.append("")

    lines.extend(
        [
            PROMPT_RULE,
            "QUESTION",
            PROMPT_RULE,
            clean_question,
            PROMPT_RULE,
            "",
            "TODAY",
            PROMPT_RULE,
            today_text,
            PROMPT_RULE,
            "",
            PROMPT_RULE,
            "URL-GROUNDED ANSWER PROMPT",
            PROMPT_RULE,
        ]
    )
    return "\n".join(lines).strip()
