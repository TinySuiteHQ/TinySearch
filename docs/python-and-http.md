# Python library and HTTP API

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
to five items and uses the configured per-item result limit. `scrape_urls`
accepts a per-call `max_tokens` budget (4,000 by default); omit an item's
scrape query or use `"*"` for page-order mode. Rendering structured evidence
into an LLM prompt is explicit, so applications can store, inspect, transform,
or budget the result first.

Direct Python calls own a short-lived scraper browser per call. The MCP and
FastAPI servers instead keep it warm between nearby requests and close it after
`browser_idle_shutdown_seconds`.

## FastAPI

The optional FastAPI app mirrors these surfaces. `POST /search` accepts the
same batch JSON contract. `POST /scrape` accepts one to five
`{ "url", "query" }` items and always returns structured per-item outcomes.
`POST /browser/navigate` and `POST /browser/act` mirror the two MCP browser
tools and return their accessibility view as `{ "result": "..." }`.
The app also exposes `/health`, `/current_datetime`, and read-only `/config`;
configuration writes require explicit environment opt-in.

## Entrypoints

- `tinysearch.search` and `tinysearch.scrape_urls`: structured Python API
- `tinysearch.get_current_datetime`: structured UTC date and time
- `tinysearch.to_prompt`: pure structured-evidence prompt renderer
- `tinysearch mcp`: stdio MCP server (also the no-argument default)
- `tinysearch serve`: Streamable HTTP MCP server
- `tinysearch.servers.fastapi_server:app`: optional FastAPI application
