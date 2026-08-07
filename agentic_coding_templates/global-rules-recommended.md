## MCP Knowledge Pack (tinysearch)

This environment uses the **tinysearch** MCP server. It exposes **three** tools.

### Available tools

**`get_current_datetime()`**

- **Input:** none.
- **Output:** `{"date_utc", "time_utc"}` in UTC.
- **When to use:** before time-sensitive research, relative-date questions (`latest`, `this year`, `last month`), or when you need to add year/month/day context to a `research` query.

**`research(query)`**

- **Input:** a single string field **`query`** only. Formulate the query for effective retrieval. You may rewrite, clarify terminology, correct spelling, expand abbreviations, add relevant temporal context, translate, or narrow the request when useful. Preserve important names, constraints, qualifiers, negations, and the user’s underlying intent.
- **Output:** a **search-grounded XML prompt** directly (not a finished article). It aggregates ranked web results, crawled page text, and chunk context. **Your job** is to answer the user from that prompt and **cite source URLs** contained in its `<result>` elements.

**`scrape_url(url, query)`**

- **Input:** **`url`** (required) and **`query`** (required, non-empty). Use a focused retrieval query describing the information to find within the page; it does not need to reproduce the user’s wording.
- **When to use:** the user supplied a specific URL, or a prior search already identified the exact page to inspect. Do **not** use this for open-ended discovery, use **`research(query)`** instead.
- **Output:** a **URL-grounded XML prompt** directly (not a finished article). It contains the page content most relevant to `query`; retrieval diagnostics are attributes on `<url_grounded_answer>`. **Your job** is to answer from that prompt and **cite the URL in its `<page>` element**.
- **Errors:** failures surface as `ValueError` with stable code prefixes: `invalid_url`, `blocked_url`, `unsupported_document`, `empty_content`, `fetch_failed`, `fetch_timeout`.

There is **no** `access_site`, `search_web`, `lite_*`, or `mode` / `max_results` on this server. Use **`get_current_datetime()`** for current UTC time, **`research(query)`** for discovery, and **`scrape_url(url, query)`** when you already know the page to read.

---

### Tool routing (most important)

#### Compound questions about “what do we use + is there something newer/better?”
Always split into two sequential steps:

1. **Codebase first**, search/read local files to find what the project actually uses. Never skip this on the assumption you already know.
2. **MCP second**, call **`research(query)`** for discovery or **`scrape_url(url, query)`** when you already have the exact page URL. Use the returned XML prompt as the evidence base.

This order is mandatory. Reversing it (or only doing the MCP half) mis-describes the project.

#### When to use codebase tools (search/read/list files)
- “What model / config / setting does this project use for X?”
- “Where is X configured / called / defined?”
- “Which version / provider / endpoint does the code target?”
- Anything answerable from files in the repo.

#### When to use `get_current_datetime()`
Use it before **`research(query)`** when:
- the question is time-sensitive or uses relative dates
- you need the current year/month/day to orient a search query

#### When to use `research(query)`
Use it when you need **up-to-date external facts** and primary sources, and you do **not** yet know which page to read:
- “Is there a newer version of X?”
- “What does the vendor doc say about Y?”
- “What are the alternatives to Z?”
- “What changed between versions?”

Formulate the query around the information needed rather than mechanically copying the user’s wording. Preserve factual constraints and intent when rewriting.

Prefer **URLs and short quotes** from the prompt text. External claims should be grounded in what `research` surfaced.

#### When to use `scrape_url(url, query)`
Use it when the **target page is already known**:
- The user pasted a URL and asked about its content.
- A prior `research` result (or codebase link) already identified the exact page.
- You need focused extraction from one doc, article, or PDF/DOCX URL, not a web-wide search.

Use **`query`** to describe the specific information you need from the page. Refine or narrow it as needed for retrieval while preserving the relevant constraints. Cite the URL inside the returned prompt’s **`<page>`** element (it may differ from the input after redirects).

#### Source hygiene
- Prefer **official docs** over blogs when the prompt includes them.
- Prefer **changelogs / release notes** for “what’s new”.
- If sources in the prompt conflict, report the conflict and cite both.

---

### Strategy with three tools

1. **Time check:** for time-sensitive or relative-date questions, call **`get_current_datetime()`** first unless you already know the current UTC date and time.
2. **Discovery first:** if you do not yet know which page to read, call **`research(query)`** once with a clear retrieval query aligned to the user’s goal.
3. **Known URL:** if the user gave a URL (or one is already identified), call **`scrape_url(url, query)`** instead of re-searching for the same page.
4. **Answer from the XML prompt**: synthesize the user’s reply from the grounded evidence and pull URLs from its `<result>` or `<page>` elements for citations.
5. If the prompt is thin on one angle, **refine `query`** and call the same tool again with a narrower follow-up, or switch tools only when the gap is “find a page” (`research`) vs “read this page” (`scrape_url`).

#### If the user gave an exact URL
Call **`scrape_url(url, query)`** with the pasted URL and a focused query describing what you need to extract. Do not route URL-only fetches through **`research(query)`** unless you also need broader discovery beyond that page.

#### If results look partial or empty
- For **`research`**, retry at most once with a **tightened or alternate `query`**.
- For **`scrape_url`**, retry at most once with a **refined `query`**.
- After two failures, say what you tried and ask for guidance.

---

## Structured approach for “what we use / something better?”

```
Step 1 – Search the codebase for the relevant config/constant/import.
Step 2 – Read the specific file(s) to confirm current value/behavior.
Step 3 – Call research(query) for discovery, or scrape_url(url, query) if you already have the doc URL.
Step 4 – Answer from the XML prompt; cite URLs in its result or page elements.
Step 5 – Synthesize: "Project uses X. Sources in the MCP prompt suggest Y. Upgrade path Z."
```

Do not emit a final recommendation until the applicable steps are done.

---

## Tool-loop prevention

- **Same tool, same args → stop.** If `research(query)` or `scrape_url(url, query)` with identical arguments returned the same class of result twice, change the query, URL, or approach—do not spam identical calls.
- **Three-strike rule.** After three consecutive tool calls with no new actionable information, pause and reassess.
- **No circular read→tool→read chains** that don’t add facts.
- **Failed MCP call.** At most one retry with a refined `query` (or a different `url` for `scrape_url`); then report attempts and limitations.
- **Progress gate.** Before each call, ask what new information it will add; if unclear, don’t call.

---

## PowerShell command guidelines (Windows environment)

When executing commands on Windows, use **PowerShell** syntax:

### Command style
- Use concise commands; avoid noisy output when possible.
- Prefer pipelines over needless intermediate variables.
- Use `--` to separate options from positional arguments that may start with `-`.

### Common patterns
```powershell
command && echo "Success" || echo "Failed"
command 2>&1 | Out-Null
Get-Content -Path "file.txt"
Set-Content -Path "file.txt" -Value "content"
Add-Content -Path "file.txt" -Value "more content"
Get-ChildItem -Recurse
Select-String -Pattern "pattern" -Path "*.py" | Select-Object -First 10
Start-Process powershell -ArgumentList "command" -Verb RunAs
```

### Error handling
- Check exit codes for critical commands.
- Use `try/catch` where failures are expected.
- Use `2>&1` when capturing stderr with stdout.

### Linter / typecheck / tests
After edits, verify quality as the project expects, for example:
```powershell
flake8 path/to/file.py
mypy path/to/file.py --ignore-missing-imports
pytest tests/ -q
```

---

## General code behaviour

- When in doubt about a service or config value, search the codebase before guessing.
- Do not assume the project matches vendor defaults, verify from source.
- For external discovery, rely on **`research(query)`** and cite URLs from the returned prompt.
- For a known page URL, rely on **`scrape_url(url, query)`** and cite the returned `url`.
