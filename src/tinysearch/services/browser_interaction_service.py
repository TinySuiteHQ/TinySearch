"""The click/type/scroll/wait action vocabulary for the `browse` primitive.

Owns exactly two things: compiling a validated action dict into a small,
self-contained JS snippet, and running a list of those snippets against an
already-open page (via `interact_and_extract`) before handing the resulting
HTML/Markdown back in the same shape `site_crawl_service.fetch_html_for_query`
uses, so it can be used as a scrape-pipeline `crawl_fn`.

Session lifecycle (which page stays open, for how long) is owned separately
by `browser_session_service`; this module only knows how to act on a page
it is handed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from tinysearch.services.site_crawl_service import (
    BOILERPLATE_EXCLUDED_TAGS,
    DEFAULT_MARKDOWN_GENERATOR_OPTIONS,
    ensure_utf8_stdio,
    get_html,
    get_markdown_fit,
    get_markdown_raw,
)
from tinysearch.telemetry import span_scope


class InvalidBrowserActionError(ValueError):
    """A requested browser action was missing required fields or malformed."""


class BrowserActionFailedError(Exception):
    """A browser action's JS ran but failed (e.g. its target selector never appeared)."""


_ALLOWED_BROWSER_ACTIONS = {"click", "type", "scroll", "wait"}


@lru_cache(maxsize=1)
def _crawl4ai_markdown_stack() -> tuple[Any, Any, Any]:
    """Import crawl4ai only when interacting; avoids heavy DLL init before embedding in MCP."""
    from crawl4ai import CrawlerRunConfig
    from crawl4ai.content_filter_strategy import BM25ContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    return CrawlerRunConfig, BM25ContentFilter, DefaultMarkdownGenerator


def _require_selector(action: Mapping[str, Any], key: str = "selector") -> str:
    value = action.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidBrowserActionError(
            f"{action.get('action')!r} action requires a non-empty {key!r}"
        )
    return value


def _positive_seconds(action: Mapping[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidBrowserActionError(f"{key!r} must be a positive number") from exc
    if number <= 0:
        raise InvalidBrowserActionError(f"{key!r} must be a positive number")
    return number


def _poll_for_selector_js(selector: str, timeout_ms: int) -> str:
    return f"""
  const selector = {json.dumps(selector)};
  const deadline = Date.now() + {timeout_ms};
  let el = null;
  while (Date.now() < deadline) {{
    el = document.querySelector(selector);
    if (el) break;
    await new Promise((r) => setTimeout(r, 100));
  }}
  if (!el) throw new Error('tinysearch: target not found: ' + selector);
""".strip("\n")


def build_action_script(
    action: Mapping[str, Any], *, default_timeout_seconds: float
) -> str:
    """Compile one validated browser action into a JS snippet body.

    Crawl4AI runs each returned string as the *body* of its own
    ``async () => { <body> }`` wrapper (one per `CrawlerRunConfig.js_code`
    entry) and awaits it, so this must NOT add another `(async () => {...})()`
    layer of its own -- doing so would let crawl4ai's wrapper return before
    the action (or its poll-for-selector loop) actually finished, and would
    turn a thrown error into a silently-dropped unhandled rejection instead
    of the `{success: False, error: ...}` crawl4ai's own try/catch produces
    and `interact_and_extract` checks. Each snippet polls for its target
    selector up to a deadline before acting. Raises
    `InvalidBrowserActionError` for an unknown action or malformed fields.
    """
    kind = action.get("action")
    if kind not in _ALLOWED_BROWSER_ACTIONS:
        raise InvalidBrowserActionError(
            f"action must be one of {sorted(_ALLOWED_BROWSER_ACTIONS)}, not {kind!r}"
        )

    if kind == "click":
        selector = _require_selector(action)
        timeout_ms = int(
            _positive_seconds(action, "timeout_seconds", default_timeout_seconds) * 1000
        )
        return (
            _poll_for_selector_js(selector, timeout_ms)
            + "\n  el.scrollIntoView({block: 'center'});\n  el.click();"
        )

    if kind == "type":
        selector = _require_selector(action)
        text = action.get("text")
        if not isinstance(text, str):
            raise InvalidBrowserActionError("'type' action requires a string 'text'")
        submit = bool(action.get("submit", False))
        timeout_ms = int(
            _positive_seconds(action, "timeout_seconds", default_timeout_seconds) * 1000
        )
        return _poll_for_selector_js(selector, timeout_ms) + f"""
  el.scrollIntoView({{block: 'center'}});
  el.focus();
  const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
  if (setter) {{ setter.call(el, {json.dumps(text)}); }} else {{ el.value = {json.dumps(text)}; }}
  el.dispatchEvent(new Event('input', {{bubbles: true}}));
  el.dispatchEvent(new Event('change', {{bubbles: true}}));
  if ({json.dumps(submit)}) {{
    el.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', code: 'Enter', bubbles: true}}));
    el.dispatchEvent(new KeyboardEvent('keyup', {{key: 'Enter', code: 'Enter', bubbles: true}}));
    const form = el.closest('form');
    if (form) {{
      if (typeof form.requestSubmit === 'function') {{ form.requestSubmit(); }} else {{ form.submit(); }}
    }}
  }}"""

    if kind == "scroll":
        selector, to, amount = action.get("selector"), action.get("to"), action.get("amount")
        provided = [value is not None for value in (selector, to, amount)]
        if sum(provided) != 1:
            raise InvalidBrowserActionError(
                "'scroll' action requires exactly one of selector, to, amount"
            )
        if selector is not None:
            if not isinstance(selector, str) or not selector.strip():
                raise InvalidBrowserActionError("'scroll' selector must be a non-empty string")
            return (
                f"  const el = document.querySelector({json.dumps(selector)});\n"
                f"  if (!el) throw new Error('tinysearch: target not found: ' + {json.dumps(selector)});\n"
                "  el.scrollIntoView({block: 'center'});"
            )
        if to is not None:
            if to not in ("top", "bottom"):
                raise InvalidBrowserActionError("'scroll' to must be 'top' or 'bottom'")
            y = "0" if to == "top" else "document.body.scrollHeight"
            return f"window.scrollTo(0, {y});"
        try:
            pixels = int(amount)
        except (TypeError, ValueError) as exc:
            raise InvalidBrowserActionError(
                "'scroll' amount must be an integer number of pixels"
            ) from exc
        return f"window.scrollBy(0, {pixels});"

    # kind == "wait"
    seconds, selector = action.get("seconds"), action.get("selector")
    if (seconds is None) == (selector is None):
        raise InvalidBrowserActionError(
            "'wait' action requires exactly one of seconds, selector"
        )
    if seconds is not None:
        delay_ms = int(_positive_seconds(action, "seconds", default_timeout_seconds) * 1000)
        return f"  await new Promise((r) => setTimeout(r, {delay_ms}));"
    if not isinstance(selector, str) or not selector.strip():
        raise InvalidBrowserActionError("'wait' selector must be a non-empty string")
    timeout_ms = int(
        _positive_seconds(action, "timeout_seconds", default_timeout_seconds) * 1000
    )
    return _poll_for_selector_js(selector, timeout_ms)


def _raise_on_action_failure(js_execution_result: Any) -> None:
    """Crawl4AI only logs a warning on a failed js_code entry; surface it as an error instead.

    ``js_execution_result`` is ``{"success": True, "results": [{"success": bool, "error": str}, ...]}``,
    one entry per `CrawlerRunConfig.js_code` script, in order -- regardless of
    the outer "success" (which just means the runner itself didn't crash).
    """
    if not isinstance(js_execution_result, Mapping):
        return
    results = js_execution_result.get("results")
    if not isinstance(results, list):
        return
    for index, script_result in enumerate(results, start=1):
        if isinstance(script_result, Mapping) and script_result.get("success") is False:
            raise BrowserActionFailedError(
                f"browser action {index} failed: {script_result.get('error') or 'unknown error'}"
            )


async def interact_and_extract(
    url: str,
    user_query: str | None,
    *,
    actions: Sequence[Mapping[str, Any]],
    bm25_threshold: float = 1.5,
    bm25_language: str = "english",
    crawler: Any,
    crawl4ai_session_id: str,
    navigate: bool,
    default_action_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run browser actions against a (possibly already-open) page, then extract it.

    Returns the same shape as `site_crawl_service.fetch_html_for_query`
    (`final_url`, `html`, `markdown_raw`, `markdown_fit`, `metadata`) so it
    can be used as a scrape-pipeline `crawl_fn`. `navigate=False` reuses the
    page already loaded under `crawl4ai_session_id` (via `js_only`) instead
    of reloading it, so state from earlier actions in the same session is
    preserved. `crawler` must be an already-started `AsyncWebCrawler`
    belonging to that same session (see `browser_session_service`).
    """
    ensure_utf8_stdio()

    CrawlerRunConfig, BM25ContentFilter, DefaultMarkdownGenerator = (
        _crawl4ai_markdown_stack()
    )

    scripts = [
        build_action_script(action, default_timeout_seconds=default_action_timeout_seconds)
        for action in actions
    ]

    config_kwargs: dict[str, Any] = {
        "verbose": False,
        "excluded_tags": BOILERPLATE_EXCLUDED_TAGS,
        "session_id": crawl4ai_session_id,
        "js_only": not navigate,
    }
    if scripts:
        config_kwargs["js_code"] = scripts
    if user_query:
        bm25_filter = BM25ContentFilter(
            user_query=user_query,
            bm25_threshold=bm25_threshold,
            language=bm25_language,
        )
        config_kwargs["markdown_generator"] = DefaultMarkdownGenerator(
            content_filter=bm25_filter,
            options=dict(DEFAULT_MARKDOWN_GENERATOR_OPTIONS),
        )
    config = CrawlerRunConfig(**config_kwargs)

    with span_scope(
        "tinysearch.browser",
        attributes={
            "tinysearch.browser.used": True,
            "tinysearch.browser.action.count": len(scripts),
        },
        operation="browser_interact",
    ) as browser_telemetry:
        result = await crawler.arun(url=url, config=config)
        browser_telemetry.complete()

    _raise_on_action_failure(getattr(result, "js_execution_result", None))

    final_url = (
        getattr(result, "redirected_url", None)
        or getattr(result, "url", None)
        or url
    )
    metadata_obj = getattr(result, "metadata", None)
    metadata = dict(metadata_obj) if isinstance(metadata_obj, dict) else {}

    return {
        "final_url": str(final_url),
        "html": get_html(result),
        "markdown_raw": get_markdown_raw(result),
        "markdown_fit": (get_markdown_fit(result) or "") if user_query else "",
        "metadata": metadata,
    }
