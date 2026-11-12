# Browser automation

`scrape_urls` is a static fetch. It cannot see content that JavaScript renders
after load, content behind a cookie interstitial, a "load more" control, or a
client-side search UI. For those pages TinySearch drives a real browser.

This costs nothing extra to install. TinySearch already scrapes pages with
Playwright and installs its Chromium for that, so the browser
tools reuse the same driver and the same browser: no second runtime, no second
browser, no child process.

Two tools: `browser_navigate` and `browser_act`, the second folding `look`,
`click`, `type`, `wait_for`, `tabs`, and `close` behind one
`action` parameter.

That split is deliberate. MCP has no way to group or nest tools -- `tools/list`
is flat and every schema is re-sent to the model on every request -- so seven
separate browser tools would dominate the server's schema. Publishing the entry
point as its own tool and folding one page session's lifecycle behind a
dispatcher keeps the whole server's schema small across five tools.

There is no separate find tool, because finding is not a sibling of clicking --
it is a filter on the result. Both tools take one `find` argument, tried as a
regex first (so `"a|b"` works directly) and falling back to a literal,
case-insensitive substring match for text that isn't valid regex -- narrowing
the return value to the matching nodes and their context instead of the whole
tree. That is what lets one call both act and report: a click that reveals a
table comes back as the table, so the agent never spends a second call
narrowing the first one's answer. An earlier version split this into `find`
and `find_regex`; that cost a wasted round trip whenever a model reached for
alternation syntax on the plain-substring parameter and got "no matches"
instead of a hint, so the two were merged.

The model reads a compact accessibility tree where each node carries a stable
ref, names one, and TinySearch acts on it with genuine browser input events:

```
browser_navigate  -> url: "...", find: "Accept"   -> "- button \"Accept all\" [ref=e79]"
browser_act       -> action: "click", target: "e79", find: "Results"
```

Nothing synthesizes DOM events or invents CSS selectors, and `click`/`type`
accept only a ref the model actually observed, never a raw selector.

Three deliberate choices:

- **No tool can execute code.** There is no `evaluate` tool, and none that
  fills forms, uploads, or drags. A page that injects instructions into its own
  rendered text has nothing dangerous to reach for, because the capability is
  absent rather than discouraged.
- **`find` is the token lever, `depth` the fallback.** Any call that returns a
  view takes `find`, cutting it to the matching nodes and their context. When no
  filter can name the target, `depth` returns a shallower but still valid tree
  rather than a truncated string -- on a large page ~700 characters versus
  ~33,000.
- **Cookies persist, sessions don't.** Set `browser_storage_state_path` and a
  consent banner accepted once is not paid for on every later navigation.
  Sessions stay isolated, so no browser profile lock is taken and concurrent
  clients do not conflict. That file is a server-side path, never exposed to a
  model or over HTTP. On a server that tells callers apart (a trusted
  identity is configured), each caller keeps their own file next to it, named
  by a salted hash, so one person's logins never reach another's browser. A
  caller without an identity, and the page fetcher's browser that serves
  everyone, use none.

To turn the tools off entirely, set `"browser_backend": "off"`; they are then
removed from the tool list rather than merely refusing to run.

## Sessions

The MCP and FastAPI servers keep the scraper browser warm between nearby
requests, then close it after `browser_idle_shutdown_seconds`. Direct Python
calls retain their short-lived, caller-owned lifecycle.

The interactive `browser_*` tools give every MCP client connection its own
browser and page, so clients sharing one server never see each other's tabs
or cookies. At most `browser_max_sessions` (default 4) are open at once; a
new caller waits up to `browser_session_wait_seconds` (default 30) for one to
close, explicitly or after its idle timeout, then gets an error. The FastAPI
`/browser/*` endpoints share one browser.

## External browser over CDP

To drive a browser you operate separately, set its Chrome DevTools Protocol
endpoint. It is used by **both** the scrape pipeline and the browser tools:

```json
{
  "browser_cdp_url": "http://browser:9222"
}
```

Server processes also accept `TINYSEARCH_BROWSER_CDP_URL`. The external browser
owns its executable, profile, proxy, and fingerprint configuration; TinySearch
does not select or install a particular browser backend.

Treat a CDP endpoint as privileged remote control of the browser. Keep it on a
private network or loopback interface, require authentication when it crosses
a host boundary, and do not expose port 9222 directly to the public internet.
When TinySearch itself runs in Docker, `localhost` refers to the TinySearch
container, so use an endpoint reachable from that container.

`browser_backend`, `browser_cdp_url`, and `browser_storage_state_path` are
operator-managed and cannot be changed through the
HTTP `PUT /config` endpoint, even when configuration writes are enabled. Set
them in the startup environment (`TINYSEARCH_BROWSER_BACKEND` and friends) or
the file selected by `TINYSEARCH_CONFIG_PATH`, then restart TinySearch. HTTP
clients can continue updating other settings by omitting these fields from
their partial update.

## Proxies

To route crawling through a proxy, set `browser_proxy_server`:

```json
{
  "browser_proxy_server": "http://gateway.proxyprovider.com:8000",
  "browser_proxy_username": "your-username",
  "browser_proxy_password": "your-password"
}
```

`http`, `https`, and `socks5` proxy URLs are supported, matching what
Playwright accepts. If your provider hands out several fixed gateway
endpoints, comma-separate them and TinySearch round-robins one per new
browser context:

```json
{
  "browser_proxy_server": "http://gw1.proxyprovider.com:8000, http://gw2.proxyprovider.com:8000"
}
```

Use `browser_proxy_bypass` for a comma-separated list of hosts that should
skip the proxy. These four fields are operator-managed like the CDP settings
above -- settable via `TINYSEARCH_BROWSER_PROXY_SERVER`,
`TINYSEARCH_BROWSER_PROXY_USERNAME`, `TINYSEARCH_BROWSER_PROXY_PASSWORD`, and
`TINYSEARCH_BROWSER_PROXY_BYPASS`, or the config file, but never over HTTP.
