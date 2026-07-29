"""Measure how many tokens TinySearch saves vs. a naive "search + fetch full page" agent.

## What this measures

Claude Code's WebSearch/WebFetch, Codex's web search, Cline's web search, and
OpenWebUI's web search all follow the same basic pattern: run a query, get a
list of snippets, then fetch one or more of those pages and hand the model
their extracted text (usually the full page, converted to markdown, with at
most a generous length cap). None of them locally rerank the fetched content
down to just the passages that answer the query, that filtering happens
inside the model's own context window, at the model's token price.

TinySearch does that filtering locally: it crawls the same pages, embeds and
reranks the chunks against the query, and returns only the top chunks plus
their source URLs.

This script isolates *exactly that difference*. For each benchmark query it:

1. Runs TinySearch's real research pipeline (search -> crawl -> hybrid rerank)
   and captures the *full, unfiltered* markdown TinySearch's own crawler
   extracted from every page it fetched, before local reranking/chunking
   throws any of it away. That's the "naive" baseline: what a tool would be
   handing the model if it fetched the same pages and pasted them in as-is.
2. Adds the token cost of the raw search-result snippets (title/url/snippet
   for every result the search backend returned), since a naive tool's
   search step returns those too.
3. Counts tokens in the actual JSON payload TinySearch returns for the same
   query, i.e. what really lands in the agent's context via the MCP tool
   result.
4. Reports the difference.

Held constant on purpose: which pages get fetched. The baseline reuses
TinySearch's own URL selection, so the number isolates the token cost of
*local compaction* (reranking + chunking), not TinySearch's search/crawl
choices. This is deliberately the more conservative (harder-to-inflate)
comparison, a real naive tool might also fetch worse pages or more of them.

This does not literally invoke Claude Code, Codex, Cline, or OpenWebUI (they
aren't callable as libraries, and several require API access this script
doesn't have). It models the fetch-and-paste behavior common to all of them.
If any of those tools caps fetched page length below what's measured here,
the real-world savings will be smaller than reported; if they fetch more
pages than TinySearch does for the same query, savings will be larger.

Runs fully locally against the search/crawl backends configured in
configs/tinysearch_config.json (same as any other TinySearch entry point).

## Usage

    python scripts/benchmark_token_savings.py
    python scripts/benchmark_token_savings.py --queries "what is a bloom filter" "explain CAP theorem"
    python scripts/benchmark_token_savings.py --json-out scripts/benchmark_token_savings.latest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from tinysearch.config import resolve_config
from tinysearch.pipelines.research import run_research_pipeline
from tinysearch.services.embedding_service import normalize_embedding_backend
from tinysearch.services.onnx_bundle_service import ensure_onnx_bundle_sync
from tinysearch.services.site_crawl_service import create_browser_crawler, crawl
from tinysearch.services.token_counter_service import token_count
from tinysearch.services.web_search_service import search

DEFAULT_QUERIES = [
    "what is the walrus operator in Python",
    "how does TCP congestion control work",
    "what is a bloom filter used for",
    "explain the CAP theorem in distributed systems",
    "what is the difference between REST and gRPC",
    "how does DNS resolution work",
    "what is a Bloom filter false positive rate",
    "how does garbage collection work in the JVM",
]


def _search_snippet_block(results: list[Any]) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"Title: {result.title}")
        lines.append(f"URL: {result.url}")
        lines.append(f"Snippet: {result.text}")
        lines.append("")
    return "\n".join(lines)


def _make_capturing_crawl_fn(capture: dict[str, str], shared_crawler: Any):
    async def _crawl_fn(
        *,
        url: str,
        encoding_name: str,
        user_query: str,
        fit_markdown_mode: str,
        fit_min_chars: int,
        bm25_threshold: float,
        bm25_language: str,
        pruning_threshold: float,
        crawler: Any = None,
    ) -> dict[str, Any]:
        result = await crawl(
            url=url,
            encoding_name=encoding_name,
            user_query=user_query,
            fit_markdown_mode=fit_markdown_mode,
            fit_min_chars=fit_min_chars,
            bm25_threshold=bm25_threshold,
            bm25_language=bm25_language,
            pruning_threshold=pruning_threshold,
            crawler=shared_crawler,
        )
        capture[url] = str(result.get("markdown_raw") or result.get("markdown") or "")
        return result

    return _crawl_fn


@dataclass
class QueryBenchmark:
    query: str
    pages_fetched: int
    naive_snippet_tokens: int
    naive_page_tokens: int
    naive_tokens: int
    tinysearch_tokens: int
    tokens_saved: int
    pct_saved: float
    elapsed_s: float


async def _benchmark_query(
    query: str,
    *,
    resolved_config: dict[str, Any],
    encoding_name: str,
    shared_crawler: Any,
) -> QueryBenchmark:
    t0 = time.perf_counter()

    raw_results = search(query, max(1, resolved_config["search_top_k"]), config=resolved_config)
    snippet_block = _search_snippet_block(raw_results)
    snippet_tokens = token_count(snippet_block, encoding_name)

    capture: dict[str, str] = {}
    result = await run_research_pipeline(
        query,
        config=resolved_config,
        search_fn=partial(search, config=resolved_config),
        crawl_fn=_make_capturing_crawl_fn(capture, shared_crawler),
    )
    payload = result.to_dict()
    tinysearch_tokens = token_count(json.dumps(payload, ensure_ascii=False), encoding_name)

    page_tokens = sum(token_count(text, encoding_name) for text in capture.values())
    naive_tokens = snippet_tokens + page_tokens
    saved = naive_tokens - tinysearch_tokens
    pct_saved = (saved / naive_tokens * 100.0) if naive_tokens else 0.0

    return QueryBenchmark(
        query=query,
        pages_fetched=len(capture),
        naive_snippet_tokens=snippet_tokens,
        naive_page_tokens=page_tokens,
        naive_tokens=naive_tokens,
        tinysearch_tokens=tinysearch_tokens,
        tokens_saved=saved,
        pct_saved=pct_saved,
        elapsed_s=time.perf_counter() - t0,
    )


async def _run(queries: list[str]) -> list[QueryBenchmark]:
    resolved_config = resolve_config(None).to_dict()
    encoding_name = str(resolved_config["encoding_name"])

    if normalize_embedding_backend(str(resolved_config["embedding_backend"])) == "onnx":
        await asyncio.to_thread(ensure_onnx_bundle_sync, str(resolved_config["embedding_model"]))

    results: list[QueryBenchmark] = []
    async with create_browser_crawler() as shared_crawler:
        for query in queries:
            print(f"[benchmark] running {query!r} ...", flush=True)
            bench = await _benchmark_query(
                query,
                resolved_config=resolved_config,
                encoding_name=encoding_name,
                shared_crawler=shared_crawler,
            )
            results.append(bench)
            print(
                f"[benchmark]   naive={bench.naive_tokens} tinysearch={bench.tinysearch_tokens} "
                f"saved={bench.tokens_saved} ({bench.pct_saved:.1f}%) "
                f"pages={bench.pages_fetched} in {bench.elapsed_s:.1f}s",
                flush=True,
            )
    return results


def _print_report(results: list[QueryBenchmark]) -> None:
    print()
    print(f"{'query':<50} {'naive':>8} {'tinysearch':>10} {'saved':>8} {'%saved':>7}")
    print("-" * 87)
    for bench in results:
        label = bench.query if len(bench.query) <= 49 else bench.query[:46] + "..."
        print(
            f"{label:<50} {bench.naive_tokens:>8} {bench.tinysearch_tokens:>10} "
            f"{bench.tokens_saved:>8} {bench.pct_saved:>6.1f}%"
        )
    print("-" * 87)

    total_naive = sum(b.naive_tokens for b in results)
    total_tinysearch = sum(b.tinysearch_tokens for b in results)
    total_saved = total_naive - total_tinysearch
    overall_pct = (total_saved / total_naive * 100.0) if total_naive else 0.0
    avg_pct = sum(b.pct_saved for b in results) / len(results) if results else 0.0

    print(
        f"{'TOTAL':<50} {total_naive:>8} {total_tinysearch:>10} {total_saved:>8} {overall_pct:>6.1f}%"
    )
    print()
    print(
        f"[benchmark] {len(results)} queries: naive baseline used {total_naive} tokens, "
        f"TinySearch used {total_tinysearch} tokens -> {overall_pct:.1f}% fewer tokens overall "
        f"({avg_pct:.1f}% average per query)."
    )


def _write_json_report(path: Path, results: list[QueryBenchmark]) -> None:
    total_naive = sum(b.naive_tokens for b in results)
    total_tinysearch = sum(b.tinysearch_tokens for b in results)
    payload = {
        "methodology": (
            "Naive baseline = raw search-result snippets + full unfiltered markdown of "
            "every page TinySearch's own crawler fetched for the query, before local "
            "reranking/chunking. TinySearch value = tokens in the actual JSON payload "
            "TinySearch returns for the same query. Page selection is held constant "
            "(same URLs) so the delta isolates local compaction, not retrieval quality."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queries": [asdict(b) for b in results],
        "totals": {
            "naive_tokens": total_naive,
            "tinysearch_tokens": total_tinysearch,
            "tokens_saved": total_naive - total_tinysearch,
            "pct_saved": (total_naive - total_tinysearch) / total_naive * 100.0 if total_naive else 0.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Queries to benchmark (default: a fixed built-in pool).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report to.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    queries = args.queries or DEFAULT_QUERIES

    results = asyncio.run(_run(queries))
    _print_report(results)
    if args.json_out:
        _write_json_report(args.json_out, results)
