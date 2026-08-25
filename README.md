# TinySearch

<!-- mcp-name: io.github.TinySuiteHQ/tinysearch -->

<p align="center">
  <a href="https://tinysuite.dev">
    <img src="assets/tinysearch-full-logo.png" alt="TinySearch" width="240" />
  </a>
</p>

<p align="center">
  <strong>Spend tokens on answers, not webpages.</strong>
</p>

<p align="center">
  TinySearch searches, crawls, and reranks the web locally, then gives your
  agent only the evidence worth putting in its context.
</p>

<p align="center">
  <a href="https://tinysuite.dev/docs/tinysearch/">Documentation</a>
  ·
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#python-library">Python</a>
  ·
  <a href="https://discord.gg/mFFKF9bf5e">Discord</a>
</p>

[![Website](https://img.shields.io/badge/tinysuite.dev-home-000000?logo=googlechrome&logoColor=white)](https://tinysuite.dev)
[![PyPI version](https://img.shields.io/pypi/v/tinysuite-search?label=pypi)](https://pypi.org/project/tinysuite-search/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/TinySuiteHQ/TinySearch?label=release)](https://github.com/TinySuiteHQ/TinySearch/releases)
[![Last Commit](https://img.shields.io/github/last-commit/TinySuiteHQ/TinySearch)](https://github.com/TinySuiteHQ/TinySearch/commits/main)
[![Docker Pulls](https://img.shields.io/docker/pulls/marcellm01/tinysearch?label=docker%20pulls)](https://hub.docker.com/r/marcellm01/tinysearch)
[![Discord](https://img.shields.io/badge/Discord-Join%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/mFFKF9bf5e)
![MCP Server](https://img.shields.io/badge/MCP-server-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-supported-009688)

TinySearch is a self-hosted web-research tool for AI agents. It searches the
web, reads the best pages, removes low-value content, and returns compact
evidence with source URLs.

Your model receives the useful passages instead of paying to process entire
webpages.

TinySearch is part of [TinySuite](https://tinysuite.dev), a suite of focused
tools designed to make agentic operations cheaper by minimizing token usage
through smart retrieval, selection, and context-management techniques.

<p align="center">
  <img src="assets/tinysearch-readme.gif" alt="TinySearch returning source-grounded web evidence to an AI agent" width="780" />
</p>

## Choose a tier

| Tier | Use it when | Entry point | Search backend |
| --- | --- | --- | --- |
| 1. Python library | You are building with TinySuite or Python | `pip install tinysuite-search` | DDGS |
| 2. One-command MCP | An MCP client should launch TinySearch for you | `uvx --from "tinysuite-search[server]" tinysearch` | DDGS |
| 3. Docker + SearXNG | You want the full self-hosted stack and HTTP MCP | `docker compose ... up -d` | Bundled SearXNG |

Tiers 1 and 2 need no search service. Tier 3 adds a dedicated SearXNG service,
persistent model storage, and a network MCP endpoint. See the
[installation guide](https://tinysuite.dev/docs/tinysearch/) for the Docker
setup.

## The expensive part of agent research is context

A search result is not yet useful evidence. Agents often have to open several
pages, ingest navigation and boilerplate, and spend paid input tokens deciding
which passages matter.

TinySearch moves that work in front of the model:

```mermaid
flowchart LR
    A[Question] --> B[Search and crawl]
    B --> C[Local hybrid reranking]
    C --> D[Compact evidence<br/>with source URLs]
    D --> E[Your agent]
```

That lowers cost in three ways:

- **Smaller model context.** Only the best-ranked evidence chunks are returned,
  within a controlled evidence budget.
- **No metered search API required by default.** TinySearch can search through
  DDGS without a paid search provider.
- **Local retrieval by default.** ONNX embeddings and hybrid reranking run on
  your machine instead of creating embedding API charges.

Search broadly. Read locally. Pay the model only for the evidence that matters.

This is retrieval, not summarization: TinySearch selects the passages worth
keeping with local BM25 and embedding rerank, it doesn't run a model over the
page to rewrite or condense it. Every returned chunk is the original page
text, unedited, so what you cite is what the page actually said. That keeps
the pipeline fast and free to run locally, at the cost of not compacting as
aggressively as a dedicated reduction model could. A learned reduction step
is a direction we may explore later; it isn't part of TinySearch today.

Actual savings depend on the pages, evidence limits, client model, and provider
pricing. TinySearch reduces the web content sent to the model; it does not
control what the client does with that evidence afterward.

<p align="center">
  <img src="assets/token-savings-benchmark.svg" alt="Benchmark: TinySearch uses 64% fewer tokens than a naive search-and-fetch agent across 8 research queries, cutting modeled input cost per 1,000 queries from $55.08 to $20.03 at $3 per million tokens" width="900" />
</p>

The cost panel uses an illustrative $3.00 per million input-token rate and
excludes search, crawling, model output, and downstream agent use.

The naive baseline isn't a strawman product, it's the same pages TinySearch
crawled for each query, fed to the model unfiltered, the way a generic
"search, then fetch the page" tool (a plain web-search-plus-fetch loop, the
kind built into most coding agents) would. Measured against the current
recommended flow (`search` then `scrape_urls`, not the deprecated
all-in-one `research` tool) and counted on the actual MCP tool-result text,
TinySearch's primary interface. Reproduce or rerun it yourself:

```bash
python scripts/benchmark_token_savings.py --json-out report.json
```

## Quick start

With [`uv`](https://docs.astral.sh/uv/) installed, add TinySearch to any MCP
client:

```json
{
  "mcpServers": {
    "tinysearch": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--from",
        "tinysuite-search[server]",
        "tinysearch"
      ]
    }
  }
}
```

The client launches TinySearch over stdio when it needs it. No repository
clone, hosted account, or paid search key is required.

Fast `search` starts without Chromium or an embedding model. The first scrape
initializes Chromium; focused scraping and the legacy `research` tool also
initialize the configured embedding model. Pre-warm both ahead of time if you
will use those workflows:

```bash
uvx --from "tinysuite-search[server]" tinysearch setup
```

<p align="center">
  <img src="assets/demo_terminal_prompt.gif" alt="TinySearch CLI setup and first run in a terminal" width="780" />
</p>

Prefer Docker, a remote MCP endpoint, or a source checkout? Follow the
[installation guide](https://tinysuite.dev/docs/tinysearch/).

## Four MCP tools

| Tool | Use it when |
| --- | --- |
| `search(items)` | You need fast, backend-ordered discovery without crawling or reranking; batch independent subquestions when useful |
| `scrape_urls(items)` | You know one to five pages; each item may use `*` for its configured clean page-order token budget |
| `get_current_datetime()` | A question depends on the current date or time |
| `research(query)` | Legacy compatibility only; deprecated in favor of `search` followed by scraping |

TinySearch deliberately stays focused. It is a retrieval layer, not another
agent, chat interface, hosted search product, or permanent web index.

See the complete [MCP tool reference](https://tinysuite.dev/docs/tinysearch/mcp-tools/)
for parameters and response contracts.

## What your agent gets

TinySearch does not spend another model call writing the final answer. The
recommended flow is `search` for lightweight discovery, then `scrape_urls` for
the pages worth reading.

Search returns structured JSON. Use one item for a simple lookup; add multiple
items only for independent subquestions or source strategies. `domains` is a
hard positive source restriction and accepts a domain plus its subdomains:

```json
{"items":[{"query":"Form 8-K Tesla","domains":["sec.gov"]}]}
```

Each search item reports its own results and compact backend attempts. A zero
result response is distinct from a blocked, unavailable, or invalid backend.
`scrape_urls` returns selected Markdown evidence and separate related-link
navigation candidates, each with independent configured token ceilings.

MCP still uses its standard JSON-RPC transport envelope, including
protocol-level errors and optional `structuredContent`. Python and FastAPI keep
their structured JSON contracts for applications that need to store, inspect,
or transform the evidence.

## How it works

1. `search` returns backend-ordered titles, URLs, previews, upstream dates, and
   backend outcomes without starting Chromium or an embedding model.
2. `scrape_urls` reads one to five known pages concurrently. Omit an item's
   query or use `"*"` to keep clean Markdown in page order within the
   configured token budget.
3. Supply a focused item query when TinySearch should chunk and hybrid-rank
   that page before returning evidence.

The deprecated MCP `research` tool retains the older all-in-one search, crawl,
and rerank pipeline for compatibility. New MCP integrations should compose
`search` with `scrape_urls` instead.

## Python library

TinySearch also works as a regular Python package:

```bash
pip install tinysuite-search
```

```python
import asyncio
from tinysearch import scrape_urls, search


async def main():
    results = await search([{"query": "Python async tasks"}])
    print(results["items"][0]["results"])

    page_url = results["items"][0]["results"][0]["url"]
    evidence = await scrape_urls([{
        "url": page_url,
        "query": "How does asyncio cancellation work?",
    }])
    print(evidence["results"])


asyncio.run(main())
```

The Python API returns stable, JSON-serializable results. `search` accepts one
to five items and uses the configured per-item result limit. `scrape_urls` accepts a per-call `max_tokens`
budget (4,000 by default); omit an item's scrape query or use `"*"` for
page-order mode. Rendering structured evidence into an LLM prompt is explicit,
so applications can store, inspect, transform, or budget the result first.

The optional FastAPI app mirrors these surfaces. `POST /search` accepts the
same batch JSON contract; `POST /research` retains its `output_format`
(`prompt` or `json`) option.
`POST /scrape` accepts one to five `{ "url", "query" }` items and always
returns structured per-item outcomes.
The app also exposes `/health`, `/current_datetime`, and read-only `/config`;
configuration writes require explicit environment opt-in.

## Search backends

TinySearch selects a web-search backend from config, so you can start with no
search service and add one later without changing code.

- `"ddgs"` (native default): queries the [`ddgs`](https://pypi.org/project/ddgs/)
  package's automatic backend selection in-process. No SearXNG deployment
  required.
- `"searxng"` (Docker default): queries a self-hosted SearXNG instance. Falls
  back to `ddgs` on backend failure unless `search_backend_fallback` is set to
  `false`.
- `"duckduckgo"`: skips SearXNG and queries `ddgs` in DuckDuckGo-only mode.
- `"auto"`: tries SearXNG, then falls back to `ddgs` on any backend failure.

Set the `BRAVE_SEARCH_API_KEY` environment variable to add Brave's official
Web Search API as a keyed fallback for the `ddgs` and `duckduckgo` backends.
Brave is only consulted when the primary call errors or returns no results.

Full key reference, SearXNG JSON-output setup, and Compose details live in the
[configuration reference](https://tinysuite.dev/docs/tinysearch/configuration/).

## External browser over CDP

TinySearch uses its bundled Playwright Chromium by default. To use a browser
that you operate separately, set its Chrome DevTools Protocol endpoint in the
config file:

```json
{
  "browser_cdp_url": "http://browser:9222"
}
```

Server processes also accept `TINYSEARCH_BROWSER_CDP_URL`. When either setting
is present, TinySearch connects through Crawl4AI instead of installing or
launching the bundled Chromium. The external browser owns its executable,
profile, proxy, and fingerprint configuration; TinySearch does not select or
install a particular browser backend.

Treat a CDP endpoint as privileged remote control of the browser. Keep it on a
private network or loopback interface, require authentication when it crosses
a host boundary, and do not expose port 9222 directly to the public internet.
When TinySearch itself runs in Docker, `localhost` refers to the TinySearch
container, so use an endpoint reachable from that container.

The CDP endpoint is operator-managed and cannot be changed through the HTTP
`PUT /config` endpoint, even when configuration writes are enabled. Set it in
the startup environment or the file selected by `TINYSEARCH_CONFIG_PATH`, then
restart TinySearch. HTTP clients can continue updating other settings by
omitting `browser_cdp_url` from their partial update.

## Semantic deduplication of evidence

Research ranks an oversampled pool of page chunks, then trims it down to the
final evidence set. Two duplicate filters run during that trim:

1. **Lexical (always on).** Token-set Jaccard drops near-identical copied text.
   Controlled by `chunk_dedupe_jaccard_threshold` (default `0.92`; lower is more
   aggressive, `1.0` disables it).
2. **Semantic (optional).** Cosine similarity over the chunk embeddings already
   computed during ranking drops syndicated, paraphrased, or reworded passages
   that carry the same information with little literal token overlap, exactly
   the duplicates Jaccard misses.

```json
{
  "chunk_semantic_dedupe_enabled": true,
  "chunk_semantic_dedupe_threshold": 0.92
}
```

- `chunk_semantic_dedupe_enabled`: turn the semantic stage on or off. Enabled
  by default.
- `chunk_semantic_dedupe_threshold`: cosine similarity (`0` to `1`) above which a
  chunk is treated as a semantic duplicate of an already-selected chunk. Lower
  values deduplicate more aggressively; keep it conservative (~`0.92`) so
  genuinely distinct-but-related evidence is not discarded. `1.0` effectively
  disables the stage.

**Design tradeoffs.** The semantic stage reuses embeddings from the ranking
step, so it adds no extra embedding calls and runs on the already-bounded
oversampled pool. It runs *after* the cheap lexical filter and compares each
candidate against the running selected output in ranked order, so the
highest-ranked member of a duplicate group is always the one kept, and per-source
quotas and fill behavior are preserved. Setting the threshold too low can merge
distinct facts that happen to be phrased similarly, which is why it defaults to a
conservative value and can be disabled entirely.

## Why TinySearch

- **No vendor in the loop.** No TinySearch account, no required API key, no
  per-request billing, no analytics service or hosted scraped-data cache. The
  infrastructure you'd otherwise pay a search API for runs on your machine.
- **Source-grounded by construction.** Every evidence chunk is the original
  page text, still attached to its originating URL, so a claim in your
  agent's answer traces back to one specific passage instead of stopping at
  "the vendor's model said this."
- **Built around token efficiency.** Page selection and passage selection
  happen locally, before content enters model context.
- **Useful without paid infrastructure.** DDGS search and local ONNX embeddings
  are the defaults.
- **Bring your own stack when needed.** SearXNG and OpenAI-compatible embedding
  providers remain optional.
- **Works where agents already work.** Use MCP over stdio, Streamable HTTP,
  Python, FastAPI, or Docker.

## Part of TinySuite

[TinySuite](https://tinysuite.dev) is a product suite built around one idea:
agents should spend tokens on useful work, not operational overhead.

Each tool focuses on a different part of the agent workflow and uses targeted
techniques to reduce unnecessary context before it reaches the model.
TinySearch handles the web-research layer by turning pages into a small,
ranked, source-grounded evidence packet.

## Documentation

The README is the product overview. Detailed setup and operational material
lives in the TinySuite documentation:

- [TinySearch overview and installation](https://tinysuite.dev/docs/tinysearch/)
- [Configuration reference](https://tinysuite.dev/docs/tinysearch/configuration/)
- [MCP tools](https://tinysuite.dev/docs/tinysearch/mcp-tools/)
- [Troubleshooting](https://tinysuite.dev/docs/tinysearch/troubleshooting/)

The repository also contains an annotated example configuration at
[`configs/tinysearch_config.json`](configs/tinysearch_config.json).

## When not to use TinySearch

TinySearch is intentionally lightweight. Use a commercial search API,
persistent crawler, or full search index when you need:

- guaranteed search coverage or an SLA
- large-scale or scheduled indexing
- long-term page storage and change history
- enterprise observability and access controls

## Development

```bash
git clone https://github.com/TinySuiteHQ/TinySearch
cd TinySearch
python -m venv .venv
source .venv/bin/activate
pip install -e ".[server]"
python -m unittest discover tests
```

TinySearch supports Python 3.12 and newer. CI tests Python 3.12, 3.13, and 3.14
across Linux, macOS, and Windows.

## Entrypoints

- `tinysearch.search` and `tinysearch.scrape_urls`: structured Python API
- `tinysearch.research`: legacy all-in-one structured Python research pipeline
- `tinysearch.get_current_datetime`: structured UTC date and time
- `tinysearch.to_prompt`: pure structured-evidence prompt renderer
- `tinysearch mcp`: stdio MCP server (also the no-argument default)
- `tinysearch serve`: Streamable HTTP MCP server
- `tinysearch.servers.fastapi_server:app`: optional FastAPI application

## Community

Questions, ideas, and bug reports are welcome:

- [Join the TinySearch Discord](https://discord.gg/mFFKF9bf5e)
- [Open a GitHub issue](https://github.com/TinySuiteHQ/TinySearch/issues)
- [Email the maintainer](mailto:hello.marcbuilds@gmail.com)

## Privacy and license

TinySearch reads public pages and returns selected excerpts to the calling
client. Search, crawling, local embeddings, and reranking can run without
sending page content to an embedding provider. If you choose an
OpenAI-compatible embedding backend, that provider receives the text sent for
vectorization.

TinySearch is available under the [MIT License](LICENSE). Downloaded model
weights remain subject to their respective model-card licenses. See
[NOTICE](NOTICE) for third-party distribution details.
