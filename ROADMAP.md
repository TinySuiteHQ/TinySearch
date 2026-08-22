# TinySearch Roadmap

## North star

> **Make TinySearch the smallest web toolkit that lets an agent reliably investigate the open web.**

TinySearch should give an agent a small set of composable primitives for discovering, reading, and navigating web information while leaving the reasoning loop in the hands of the model using it.

The goal is not to one-shot a research task or predict everything the agent will need up front. The agent should be able to search, inspect evidence, change direction, follow up, and escalate to browser interaction when normal retrieval is insufficient.

The roadmap is intentionally ordered around capability rather than tool count. New features should make web investigation more reliable while keeping the public surface small, composable, and model-driven.

## Product principles

- **Keep the primitive set small.** Add capabilities when they unlock a distinct part of web investigation, not just because a larger web platform exposes them.
- **Let the model drive the loop.** Search, read, interact, reason, and repeat should remain composable rather than hidden behind one monolithic research call.
- **Browser interaction is a retrieval primitive.** Clicking, scrolling, waiting, or typing is useful when it helps an agent reach information that cannot otherwise be retrieved. TinySearch does not need to become a general-purpose browser automation framework.
- **Prefer existing infrastructure over reimplementing it.** Compose proven search, crawling, browser, and model components such as SearXNG, DDGS, Crawl4AI, Playwright, and external CDP browsers rather than replacing them.
- **Token efficiency supports usability; it does not define it.** Filtering, ranking, and bounded outputs should reduce unnecessary context without preventing the agent from asking for more evidence or changing course.
- **Stay useful without an internal LLM.** The core toolkit should remain directly usable by an external agent. Higher-level delegated research can be layered on later without replacing the primitives underneath it.
- **Measure before adding autonomy.** Higher-level automation should be evaluated against the underlying primitives rather than assumed to be better because it is more agentic.
- **Do not roadmap by competitor checklist.** Parallel agents, source-specific adapters, caches, monitors, or platform features should exist only when real usage demonstrates a need.

## Current shape

The core web-investigation loop is intentionally simple:

```text
current time  -> ground time-sensitive questions
search        -> discover candidate sources
scrape        -> read known sources
browser       -> interact when normal retrieval is insufficient
```

The exact tools can evolve, but new features should make this loop more reliable without turning TinySearch into an all-in-one search platform, crawler, browser automation suite, or autonomous research agent by default.

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