# Search backends

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
[configuration reference](https://tinysuite.dev/docs/tinysearch/#search-backend).
