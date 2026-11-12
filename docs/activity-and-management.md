# Activity and management views

TinySearch records nothing about what it searched unless you ask it to.
`TINYSEARCH_ACTIVITY` turns recording on:

| Mode | Kept |
| --- | --- |
| `off` (default) | Nothing. OpenTelemetry stays payload-free either way. |
| `aggregate` | Per operation: outcome, latency, error code. Per domain: how often it appeared in search results (*discovered*) or was actually fetched, and how often fetching it failed. No queries, no URLs, no callers. |
| `history` | Aggregate, plus each caller's own queries and URLs, readable and deletable only by that caller. Activity from a caller without a trusted identity stays aggregate. |
| `content` | History, plus the exact text each search, fetch and browser action returned to the caller: what the model was shown, never whole pages. The tools' own limits (`scrape_max_tokens`, `browser_response_char_budget`) decide its size, so nothing the model saw is cut. |

Records older than `TINYSEARCH_ACTIVITY_RETENTION_DAYS` (default 7) are
deleted, and only the newest `TINYSEARCH_ACTIVITY_MAX_EVENTS` (default 5000)
are kept. Callers are stored as salted hashes. Everything lives in one SQLite
file, `TINYSEARCH_ACTIVITY_DB` (default: `activity.db` in the user data
directory). On macOS and Linux it, and saved browser cookies, are created
owner-only (`0600`), in a folder created `0700`; on Windows the per-user
`AppData` folder already keeps them private.

## Management views

`TINYSEARCH_MANAGEMENT=on` exposes these records to a client UI, such as
TinyWebUI's MCP panel, as MCP resources following the `tinysuite.manage/1`
contract (manifest `tinysearch://manage/manifest`, media type
`application/vnd.tinysuite.manage+json`):

- **Overview, Domains, Blocking rules** (instance-wide): health and counts,
  discovered versus fetched domains, configured `blocked_domains`, and the
  built-in SSRF rules. A search item's own `domains` narrow that one search;
  they are not a blocklist.
- **My searches, My pages** (per caller): that caller's own history in
  `history` or `content` mode. Each entry in My searches opens to its query,
  URLs and outcome, and in `content` mode to what the model was shown. A
  `clear_search_history` tool, added only in those modes, deletes it all.

## Who may read what

- A local stdio client is the operator and sees everything.
- Over HTTP, the instance views need
  `Authorization: Bearer <TINYSEARCH_MANAGEMENT_TOKEN>` (24+ characters).
- Over HTTP, a caller's own views need a trusted identity: the
  `TINYSEARCH_TRUSTED_USER_HEADER` header, believed only from peers in
  `TINYSEARCH_TRUSTED_PROXY_CIDRS` that also send
  `X-TinySearch-Proxy-Token: <TINYSEARCH_TRUSTED_PROXY_TOKEN>` (at least 32
  characters, required with the header), checked on every message. The
  address alone does not prove the proxy sent a request: another container
  on that network, or a local program when TinySearch listens on loopback,
  could otherwise name any user. A request without the right token is treated
  as a caller with no identity: its activity stays aggregate.

Management over HTTP with neither a token nor a trusted identity refuses to
start.
