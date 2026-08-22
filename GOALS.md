# TinySearch Goals

## North star

> **Make TinySearch the smallest web toolkit that lets an agent reliably investigate the open web.**

TinySearch should give an agent a small set of composable primitives for discovering, reading, and navigating web information while leaving the reasoning loop in the hands of the model using it.

The goal is not to one-shot a research task or predict everything the agent will need up front. The agent should be able to search, inspect evidence, change direction, follow up, and escalate to browser interaction when a page cannot be understood through a normal crawl.

## What this means

- **Keep the primitive set small.** Add capabilities when they unlock a distinct part of web investigation, not just because a larger web platform exposes them.
- **Let the model drive the loop.** Search, read, interact, reason, and repeat should remain composable rather than hidden behind one monolithic research call.
- **Browser interaction is a retrieval primitive.** Clicking, scrolling, waiting, or typing is useful when it helps an agent reach information that cannot otherwise be retrieved. TinySearch does not need to become a general-purpose browser automation framework.
- **Prefer existing infrastructure over reimplementing it.** TinySearch should compose proven search, crawling, browser, and model components rather than try to replace them.
- **Token efficiency supports usability; it does not define it.** Filtering, ranking, and bounded outputs should reduce unnecessary context without preventing the agent from asking for more evidence or changing course.
- **Stay useful without an internal LLM.** The core toolkit should remain directly usable by an external agent. Higher-level delegated research can be layered on later without replacing the primitives underneath it.

## Current shape

The core web-investigation loop is intentionally simple:

```text
current time  -> ground time-sensitive questions
search        -> discover candidate sources
scrape        -> read known sources
browser       -> interact when normal retrieval is insufficient
```

The exact tools can evolve, but new features should make this loop more reliable without turning TinySearch into an all-in-one search platform, crawler, browser automation suite, or autonomous research agent by default.
