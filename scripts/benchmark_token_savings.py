"""Measure how many tokens TinySearch saves vs. a naive "search + fetch full page" agent.

## What this measures

Claude Code's WebSearch/WebFetch, Codex's web search, Cline's web search, and
OpenWebUI's web search all follow the same basic pattern: run a query, get a
list of snippets, then fetch one or more of those pages and hand the model
their extracted text (usually the full page, converted to markdown, with at
most a generous length cap). None of them locally rerank the fetched content
down to just the passages that answer the query, that filtering happens
inside the model's own context window, at the model's token price.

TinySearch does that filtering locally, via the two tools it recommends
agents actually use: `search` for fast backend-ordered discovery, then
`scrape_urls` to fetch and hybrid-rerank the pages worth reading. This
benchmark measures that recommended path.

This script isolates *exactly that difference*. For each benchmark query it:

1. Calls TinySearch's real `search` step and counts the tokens in the raw
   search-result snippets (title/url/snippet for every result returned),
   since a naive tool's search step returns those too.
2. Picks the top `--pages-per-query` results (an agent choosing which pages
   are worth opening) and scrapes each with TinySearch's real `scrape_urls`
   pipeline, passing the benchmark query as each item's query so the same
   hybrid rerank a real call would use actually runs.
3. While scraping, captures the *full, unfiltered* markdown TinySearch's own
   crawler extracted from each page, before local reranking/chunking throws
   any of it away. That's the "naive" baseline: what a tool would be handing
   the model if it fetched the same pages and pasted them in as-is.
4. Counts tokens in the actual XML text an MCP client receives as the
   `scrape_urls` tool-result for those pages -- what really lands in the
   agent's context over MCP, TinySearch's primary interface. (The Python
   library and FastAPI JSON contracts carry additional score/rank metadata
   the MCP rendering strips, so a caller reading raw JSON will see a
   somewhat larger number than this benchmark reports.)
5. Reports the difference.

Held constant on purpose: which pages get fetched. The baseline reuses
TinySearch's own top-ranked search results, so the number isolates the
token cost of *local compaction* (reranking + chunking), not TinySearch's
search/crawl choices. This is deliberately the more conservative
(harder-to-inflate) comparison; a real naive tool might also fetch worse
pages or more of them.

This does not literally invoke Claude Code, Codex, Cline, or OpenWebUI (they
aren't callable as libraries, and several require API access this script
doesn't have). It models the fetch-and-paste behavior common to all of them.
If any of those tools caps fetched page length below what's measured here,
the real-world savings will be smaller than reported; if they fetch more
pages than this benchmark does for the same query, savings will be larger.

Runs fully locally against the search/crawl backends configured in
configs/tinysearch_config.json (same as any other TinySearch entry point).

## Usage

    python scripts/benchmark_token_savings.py
    python scripts/benchmark_token_savings.py --queries "what is a bloom filter" "explain CAP theorem"
    python scripts/benchmark_token_savings.py --pages-per-query 5
    python scripts/benchmark_token_savings.py --json-out scripts/benchmark_token_savings.latest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from tinysearch.config import resolve_config
from tinysearch.pipelines.scrape import run_scrape_pipeline
from tinysearch.results import public_chunk, public_link, result_envelope
from tinysearch.services.embedding_service import normalize_embedding_backend
from tinysearch.services.grounded_prompt_service import format_url_grounded_answers
from tinysearch.services.onnx_bundle_service import ensure_onnx_bundle_sync
from tinysearch.services.site_crawl_service import (
    create_browser_crawler,
    fetch_html_for_query,
)
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

DEFAULT_PAGES_PER_QUERY = 3


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
        user_query: str | None,
        bm25_threshold: float,
        bm25_language: str,
        crawler: Any = None,
    ) -> dict[str, Any]:
        result = await fetch_html_for_query(
            url=url,
            user_query=user_query,
            bm25_threshold=bm25_threshold,
            bm25_language=bm25_language,
            crawler=crawler or shared_crawler,
        )
        capture[url] = str(result.get("markdown_raw") or "")
        return result

    return _crawl_fn


async def _scrape_one(
    url: str,
    query: str,
    *,
    max_tokens: int,
    config: dict[str, Any],
    crawler: Any,
    crawl_fn: Any,
) -> dict[str, Any]:
    """Run one scrape_urls-equivalent call, mirroring core.scrape_urls's own envelope."""
    try:
        scrape_result = await run_scrape_pipeline(
            url,
            query,
            max_tokens=max_tokens,
            include_metadata=True,
            config=config,
            crawler=crawler,
            crawl_fn=crawl_fn,
        )
    except Exception as exc:  # noqa: BLE001 - mirrors core.scrape_urls's per-item error handling
        return {
            "url": url,
            "query": query,
            "status": "error",
            "error": {"code": type(exc).__name__, "message": str(exc)},
        }
    source = {
        "id": "1",
        "title": scrape_result.title,
        "url": scrape_result.url,
        "metadata": scrape_result.metadata or {},
        "chunks": [
            public_chunk(chunk, rank=rank)
            for rank, chunk in enumerate(scrape_result.chunks, start=1)
        ],
        "links": [
            public_link(link, rank=rank)
            for rank, link in enumerate(scrape_result.links, start=1)
        ],
    }
    envelope = result_envelope(
        operation="scrape",
        status="ok",
        query=scrape_result.query,
        retrieved_at=scrape_result.retrieved_at,
        sources=[source],
        stats={
            "content_tokens": scrape_result.content_tokens,
            "truncated": scrape_result.truncated,
        },
    )
    return {"status": "ok", "result": envelope}


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


async def _search_with_retry(
    query: str,
    limit: int,
    *,
    config: dict[str, Any],
    attempts: int = 3,
    backoff_s: float = 5.0,
) -> list[Any]:
    """Retry a zero-result search before trusting it.

    The `ddgs` backend can return an empty result list for a *transient*
    rate limit rather than raising, indistinguishable at the call site from
    a genuine zero-result query. Silently treating that as "0 naive tokens"
    corrupts the benchmark (it looks like an artificially perfect score).
    Raises loudly instead of returning a number nobody should trust.
    """
    for attempt in range(1, attempts + 1):
        results = search(query, limit, config=config)
        if results:
            return results
        if attempt < attempts:
            print(
                f"[benchmark]   search returned 0 results for {query!r} "
                f"(attempt {attempt}/{attempts}), retrying in {backoff_s:.0f}s ...",
                flush=True,
            )
            await asyncio.sleep(backoff_s)
    raise RuntimeError(
        f"search() returned 0 results for {query!r} after {attempts} attempts. "
        "This is almost certainly a transient backend rate limit (not a genuine "
        "zero-result query) -- do not trust or publish this run's numbers. Wait "
        "and re-run."
    )


async def _benchmark_query(
    query: str,
    *,
    resolved_config: dict[str, Any],
    encoding_name: str,
    pages_per_query: int,
    shared_crawler: Any,
) -> QueryBenchmark:
    t0 = time.perf_counter()

    raw_results = await _search_with_retry(
        query, max(1, resolved_config["search_top_k"]), config=resolved_config
    )
    snippet_block = _search_snippet_block(raw_results)
    snippet_tokens = token_count(snippet_block, encoding_name)

    selected_urls = [result.url for result in raw_results[: max(1, pages_per_query)]]

    capture: dict[str, str] = {}
    crawl_fn = _make_capturing_crawl_fn(capture, shared_crawler)
    scrape_max_tokens = int(resolved_config["scrape_max_tokens"])

    items = await asyncio.gather(
        *(
            _scrape_one(
                url,
                query,
                max_tokens=scrape_max_tokens,
                config=resolved_config,
                crawler=shared_crawler,
                crawl_fn=crawl_fn,
            )
            for url in selected_urls
        )
    )
    # `format_url_grounded_answers` is the actual XML text an MCP client
    # receives as the scrape_urls tool-result (see servers/mcp_server.py) --
    # that's the real transport for TinySearch's primary interface, and it's
    # meaningfully leaner than the raw JSON envelope (no score breakdowns,
    # ranks, or schema scaffolding), so it's the number reported here.
    tinysearch_tokens = token_count(
        format_url_grounded_answers(results=items), encoding_name
    )

    # Document URLs (PDF/DOCX) bypass the HTML crawl_fn this benchmark hooks,
    # so they won't appear in `capture`; treated as 0 naive page tokens
    # (rare for these queries, and only ever undercounts the naive baseline).
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


async def _run(queries: list[str], pages_per_query: int) -> list[QueryBenchmark]:
    resolved_config = resolve_config(None).to_dict()
    encoding_name = str(resolved_config["encoding_name"])

    if normalize_embedding_backend(str(resolved_config["embedding_backend"])) == "onnx":
        await asyncio.to_thread(ensure_onnx_bundle_sync, str(resolved_config["embedding_model"]))

    results: list[QueryBenchmark] = []
    async with create_browser_crawler() as shared_crawler:
        for index, query in enumerate(queries):
            if index > 0:
                # Space out searches; back-to-back calls are what tripped
                # the ddgs backend's rate limit during development.
                await asyncio.sleep(3.0)
            print(f"[benchmark] running {query!r} ...", flush=True)
            bench = await _benchmark_query(
                query,
                resolved_config=resolved_config,
                encoding_name=encoding_name,
                pages_per_query=pages_per_query,
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
            "Naive baseline = raw search-result snippets (search step) + full "
            "unfiltered markdown of the top-ranked pages scrape_urls actually "
            "fetched for the query, before local reranking/chunking. TinySearch "
            "value = tokens in the XML text an MCP client receives as the "
            "scrape_urls tool-result for the same pages and query (TinySearch's "
            "primary interface; the Python/FastAPI JSON contracts carry extra "
            "score/rank metadata the MCP rendering strips, so raw-JSON callers "
            "see a somewhat larger number than this). Page selection is held "
            "constant (same URLs, same rank order search returned) so the delta "
            "isolates local compaction, not retrieval quality."
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
        "--pages-per-query",
        type=int,
        default=DEFAULT_PAGES_PER_QUERY,
        help=(
            "How many top-ranked search results to scrape per query, modeling "
            "how many pages an agent chooses to open (default: "
            f"{DEFAULT_PAGES_PER_QUERY}; scrape_urls itself caps a batch at 5)."
        ),
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

    results = asyncio.run(_run(queries, args.pages_per_query))
    _print_report(results)
    if args.json_out:
        _write_json_report(args.json_out, results)
