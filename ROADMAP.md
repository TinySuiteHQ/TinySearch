# TinySearch Roadmap

TinySearch's north star is simple:

> **Make TinySearch the smallest web toolkit that lets an agent reliably investigate the open web.**

The roadmap is intentionally ordered around capability rather than tool count. New features should make web investigation more reliable while keeping the public surface small, composable, and model-driven.

## Principles

- Keep the primitive set small.
- Let the calling model drive the investigation loop by default.
- Treat browser interaction as a retrieval mechanism, not general-purpose automation.
- Prefer existing infrastructure such as Crawl4AI, Playwright, SearXNG, and external CDP browsers over rebuilding them.
- Stay useful without an internal LLM.
- Add higher-level automation only when it can be measured against the underlying primitives.
- Do not add parallel agents, source-specific adapters, or platform features simply to match competitors.

## Phase 1 — Complete the web primitives

### Browser interaction — #53

Add a narrow browser retrieval primitive for pages that cannot be understood through a normal scrape.

Initial actions should remain small:

- click
- type
- scroll
- wait

The resulting page should flow back through TinySearch's existing extraction and relevance pipeline instead of returning an unbounded browser/DOM dump.

### Authenticated browser sessions — #59

Add first-class support for already-authenticated browser sessions using Crawl4AI-supported session/profile mechanisms and external CDP browsers.

TinySearch should be able to operate inside sites where the user has already logged in without owning credentials or implementing login flows itself.

This should support use cases such as authenticated documentation, private dashboards, X/YouTube/GitHub sessions, and other sites where normal anonymous retrieval is insufficient.

Credential storage and account automation remain out of scope.

## Phase 2 — Measure before adding more autonomy

### Web-investigation evaluation harness — #58

Build a small, versioned benchmark that measures whether TinySearch actually retrieves the right evidence.

Track things such as:

- authoritative sources found
- required evidence found or missed
- search/scrape/browser calls
- browser escalations
- returned tokens
- latency
- failures/timeouts

This should become the basis for deciding whether later agentic features improve the product or merely add complexity.

## Phase 3 — Optional delegated investigation

### `investigate()` worker — #54

Add an optional bounded research loop driven by a cheap/local/BYO model.

The worker should use the same primitives available to external callers:

```text
SEARCH
SCRAPE
BROWSE
DONE
```

Its job is evidence acquisition and compression, not replacing the caller's final reasoning model.

The primitive tools must remain independently usable without any internal model configured.

## Phase 4 — Research continuity

### Persistent research sessions — #55

Allow investigations to continue across calls while preserving only bounded, web-specific state such as:

- sources already inspected
- selected evidence
- unresolved questions
- browser-session references
- retrieval timestamps

This is research continuity, not a general memory system.

### Delta-aware reruns — #56

Avoid repeating expensive work for previously inspected sources that have not materially changed.

Prefer lightweight revalidation using:

- `ETag`
- `Last-Modified`
- normalized content fingerprints

Only changed or new evidence should require expensive downstream processing by default.

## Phase 5 — Adaptive retrieval

### Adaptive escalation — #57

Choose the cheapest reliable retrieval path before escalating:

```text
search/snippet
    ↓
direct scrape
    ↓
browser-assisted retrieval
    ↓
optional investigate() worker
```

Routing should use observable signals and remain inspectable in traces. Explicit primitive calls should always remain available.

## Later — only when evidence justifies it

Potential future work includes:

- source-aware handling for structured or difficult sources such as GitHub, Reddit, X, YouTube, PDFs, and documentation sites;
- parallel research workers for genuinely independent, latency-bound branches;
- richer research-state inspection and export;
- additional search/browser/model backends where real users require them.

These are intentionally not committed roadmap items yet. They should be added only when real usage or evaluation data demonstrates that the core primitives cannot solve the problem cleanly.

## What TinySearch is not trying to become

TinySearch is not trying to be:

- a full browser automation/RPA framework;
- a hosted search platform;
- a general-purpose crawler/indexer;
- a mandatory autonomous research agent;
- a multi-agent framework;
- a collection of brittle site-specific scrapers.

The intended end state is still small:

```text
current time  -> ground time-sensitive questions
search        -> discover candidate sources
scrape        -> read known sources
browser       -> navigate when normal retrieval is insufficient
investigate   -> optionally delegate iterative evidence gathering
```

Everything else should make that loop more reliable, cheaper, or easier to use.