# Fallback fetcher

`scrape_urls` reads a page with its own Chromium. Some pages don't come back
readable that way: the server answers with an error status, or the render holds
almost no text. By default TinySearch reports those cases and moves on. You can
also give it a second fetch path: an HTTP service you run, which TinySearch
asks for the page when its own read falls short.

TinySearch defines the contract below. What runs behind it -- a separate
rendering service, a hosted fetch API, or anything else that speaks the
contract -- is your choice, and it is off unless you configure it.

## When it is used

After its own render, TinySearch asks the fallback fetcher for an HTML page
when:

- the page answered **HTTP 403, 429, or 503** (forbidden, too many requests,
  temporarily unavailable), or
- the render has **fewer than `scrape_fallback_min_words` visible words**
  (default 30). Words are counted from the page text a reader sees; the
  accessibility-outline markup around it is not counted. `0` limits this to
  pages with no text at all.

PDF and DOCX URLs never use it. Pages that read normally never use it.

## What happens next

| Situation | Result |
| --- | --- |
| The fetcher returns a page with enough text | That page is used: its text, title, links and final URL |
| The original answered 403, 429 or 503 and the fetcher fails, isn't configured, or returns another such page | The item fails with `fetch_unavailable` |
| The original answered 200 with little text and the fetcher fails or returns little text | The original page is used as before |

`fetch_unavailable` applies whether or not a fallback fetcher is configured: an
error page's text is never evidence about the page that was asked for, so
TinySearch no longer returns it as if it were.

The final URL the fetcher reports goes through the same URL safety checks and
`blocked_domains` rules as any redirect.

## Settings

| Key | Default | Meaning |
| --- | --- | --- |
| `scrape_fallback_fetcher_url` | `""` (off) | The service's `http` or `https` URL |
| `scrape_fallback_fetcher_timeout_seconds` | `60` | How long the service may take; sent to it as `timeout_seconds` |
| `scrape_fallback_min_words` | `30` | Visible-word threshold for a short page; `0` = only empty pages |

`scrape_fallback_fetcher_url` is operator-managed like the browser settings:
set it in the config file or with `TINYSEARCH_SCRAPE_FALLBACK_FETCHER_URL`. It
cannot be changed through `PUT /config`, and `GET /config` masks it, since the
URL may carry a credential.

```json
{
  "scrape_fallback_fetcher_url": "http://127.0.0.1:8002/fetch",
  "scrape_fallback_fetcher_timeout_seconds": 60,
  "scrape_fallback_min_words": 30
}
```

## Contract

TinySearch sends one request per page:

```http
POST <scrape_fallback_fetcher_url>
Content-Type: application/json

{"url": "https://example.com/page", "timeout_seconds": 60}
```

A successful answer is a 2xx JSON object:

```json
{
  "html": "<html>...</html>",
  "final_url": "https://example.com/page",
  "status": 200
}
```

- `html` (required): the page's HTML.
- `final_url` (optional): the URL after redirects; defaults to the requested URL.
- `status` (optional): the HTTP status the page answered with.

Any other answer -- a non-2xx status, invalid JSON, no `html` string, a
response over 20 MB, or no answer within the timeout -- counts as a failed
fallback.

## Safety

The returned HTML is rendered in TinySearch's Chromium with JavaScript turned
off and every network request aborted, then read the same way as a normal
scrape. The service can supply page content, but nothing in that content runs
or loads anything.

Run the service on a private network or loopback interface. TinySearch sends it
every URL that triggers a fallback, so it sees which pages your agents read.
