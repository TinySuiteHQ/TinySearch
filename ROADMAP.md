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
- **Prefer existing infrastructure over reimplementing it.** Compose proven search, crawling, browser, and model components such as SearXNG, DDGS, Playwright, and external CDP browsers rather than replacing them.
- **Token efficiency supports usability; it does not define it.** Filtering, ranking, and bounded outputs should reduce unnecessary context without preventing the agent from asking for more evidence or changing course.
- **No internal research loop.** TinySearch never runs its own model-driven research loop or delegates investigation to an internal LLM. The calling model always decides what to search, read, and do next; TinySearch supplies the primitives and the evidence.
- **Measure before adding features.** Changes to retrieval should be evaluated against the existing primitives rather than assumed to be better because they are more sophisticated.
- **Do not roadmap by competitor checklist.** Source-specific adapters, caches, monitors, or platform features should exist only when real usage demonstrates a need.

## Current shape

The core web-investigation loop is intentionally simple:

```text
current time  -> ground time-sensitive questions
search        -> discover candidate sources
scrape        -> read known sources
browser       -> interact when normal retrieval is insufficient
```

The exact tools can evolve, but new features should make this loop more reliable without turning TinySearch into an all-in-one search platform, crawler, browser automation suite, or autonomous research agent.

## Phase 1 — Complete the web primitives

### Browser interaction — #53

Add a narrow browser retrieval primitive for pages that cannot be understood through a normal scrape.

Initial actions should remain small:

- click
- type
- scroll
- wait

The resulting page should flow back through TinySearch's existing extraction and relevance pipeline instead of returning an unbounded browser/DOM dump.

## Phase 2 — Measure retrieval quality

### Evidence check for the benchmark — #58

Extend the existing token-savings benchmark so it measures whether TinySearch keeps the evidence that answers each query, not only how many tokens it returns.

Each benchmark case lists the evidence a good answer must contain. Each run records whether that evidence was in the fetched pages and whether it survived into TinySearch's output, along with latency and failures. The resulting **evidence retention** sits next to token savings, so a change that saves tokens by dropping the answer shows up as a regression.

This should become the basis for deciding whether a change improves retrieval or merely adds complexity.

## Phase 3 — Faster fetching

### Escalating fetch — #57

Fetch each page the cheapest way that works, instead of starting with a full Chromium page load:

```text
plain HTTP fetch
    ↓ if the response isn't usable
full Chromium page load
    ↓ if the page is still unreadable
operator's fallback fetcher
```

Most pages ship their text in the HTML and never need their JavaScript run. Escalation uses deterministic signals (blocked status codes, too little readable text, pages built by JavaScript), stays inside a single `scrape_urls` call, and is visible in traces. Every tier reads text through the same extractor, so output looks the same whichever tier served it.

The evidence check (#58) decides whether this ships on by default: scrapes should get faster without losing evidence.

## Later — only when evidence justifies it

Potential future work includes:

- source-aware handling for structured or difficult sources such as code hosts, forums, social platforms, video sites, PDFs, and documentation sites;
- additional search/browser/model backends where real users require them.
- a guided sign-in for sites the user chooses (#59). Today an operator can give TinySearch a signed-in browser through `browser_storage_state_path` (a saved cookie file) or `browser_cdp_url` (their own browser), on their own decision and responsibility.

These are intentionally not committed roadmap items yet. They should be added only when real usage or evaluation data demonstrates that the core primitives cannot solve the problem cleanly.

## What TinySearch is not trying to become

TinySearch is not trying to be:

- a full browser automation/RPA framework;
- a hosted search platform;
- a general-purpose crawler/indexer;
- an autonomous research agent, optional or otherwise;
- a multi-agent framework;
- a collection of brittle site-specific scrapers.

The intended end state is still small:

```text
current time  -> ground time-sensitive questions
search        -> discover candidate sources
scrape        -> read known sources
browser       -> navigate when normal retrieval is insufficient
```

The research loop belongs to the calling model.

Everything else should make that loop more reliable, cheaper, or easier to use.