"""Turn a Playwright ARIA snapshot into plain reading text without losing any of it.

The scrape path reads a page as an ARIA snapshot: a YAML outline such as

    - navigation:
      - list:
        - listitem:
          - link "Home":
            - /url: /

On real pages half or more of its tokens are outline syntax rather than page
text: link targets (``/url:``), role keywords, wrapper nodes with no text,
icon-font glyphs, and children that repeat their parent's accessible name.
All of it is chunked, embedded, and in page-order mode returned to the model.

`clean_snapshot` keeps every visible string and removes only that outline
overhead. Its rules, all language-neutral (role names are always English,
page text is never inspected beyond the checks below):

- ``/url`` values are dropped. Links are extracted separately from the HTML,
  so the target adds nothing to the text.
- Role keywords, attributes, indentation, and YAML quoting are dropped.
  Headings become ``#`` lines, list items ``- `` lines, and table rows
  ``| a | b |`` lines.
- A string made only of whitespace, zero-width, or private-use characters
  (icon fonts) is dropped. Such characters inside real text are kept.
- A node's children are skipped only when every string in them already
  appears inside the node's own name, which is how browsers compute the
  accessible name of links, headings, and cells.
- Consecutive identical lines are collapsed to one.

Anything the parser does not recognise makes `clean_snapshot` return the
snapshot unchanged, so an unexpected Playwright format costs tokens rather
than text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Whitespace, zero-width and direction marks, word joiner, BOM, and the Unicode
# private-use areas, where icon fonts put their glyphs. Only a string made
# entirely of these is dropped.
_NON_TEXT_ONLY_RE = re.compile(
    r"^[\s ​-‏⁠﻿-\U000f0000-\U0010ffff]*$"
)
_ROLE = r"[a-z][\w-]*"
_QUOTED_NAME = r'"(?:[^"\\]|\\.)*"'
_SLASHED_NAME = r"/.*?/"
_ATTRS = r"(?: \[[^\]]*\])*"
_NODE_HEADER_RE = re.compile(
    rf"^(?P<role>{_ROLE})(?: (?P<quoted>{_QUOTED_NAME})| (?P<slashed>{_SLASHED_NAME}))?"
    rf"(?P<attrs>{_ATTRS})$"
)
_ENTRY_RE = re.compile(
    rf"^(?P<header>/[\w-]+|{_ROLE}(?: {_QUOTED_NAME}| {_SLASHED_NAME})?{_ATTRS})"
    r"(?:(?P<colon>:)(?: (?P<value>.*))?)?$"
)
_LEVEL_RE = re.compile(r"\[level=(\d+)\]")
_CELL_ROLES = frozenset({"cell", "gridcell", "columnheader", "rowheader"})


class SnapshotFormatError(ValueError):
    """The snapshot has a shape this module does not recognise."""


@dataclass
class _Node:
    role: str
    name: str = ""
    value: str = ""
    attrs: str = ""
    children: list[_Node] = field(default_factory=list)


def _split_quoted(text: str) -> tuple[str, str]:
    """Split a leading YAML-quoted scalar from the rest of ``text``."""
    if text.startswith("'"):
        index = 1
        while True:
            end = text.find("'", index)
            if end == -1:
                raise SnapshotFormatError(text)
            if text[end + 1 : end + 2] == "'":
                index = end + 2
                continue
            return text[1:end].replace("''", "'"), text[end + 1 :]
    match = re.match(_QUOTED_NAME, text)
    if match is None:
        raise SnapshotFormatError(text)
    return json.loads(match.group(0)), text[match.end() :]


def _scalar(text: str) -> str:
    text = text.strip()
    if text[:1] in {"'", '"'}:
        value, rest = _split_quoted(text)
        if rest.strip():
            raise SnapshotFormatError(text)
        return value
    return text


def _parse_entry(body: str) -> _Node:
    """Parse one ``- ...`` line, without its leading dash, into a node."""
    if body[:1] in {"'", '"'}:
        header, rest = _split_quoted(body)
        rest = rest.strip()
        if rest in {"", ":"}:
            value = ""
        elif rest.startswith(": "):
            value = _scalar(rest[2:])
        else:
            raise SnapshotFormatError(body)
    else:
        match = _ENTRY_RE.match(body)
        if match is None:
            raise SnapshotFormatError(body)
        header = match["header"]
        value = _scalar(match["value"]) if match["value"] is not None else ""
    if header.startswith("/"):
        return _Node(role=header, value=value)
    match = _NODE_HEADER_RE.match(header)
    if match is None:
        # A quoted entry that is not a node header is a bare text item.
        return _Node(role="text", value=header)
    if match["quoted"]:
        name = json.loads(match["quoted"])
    else:
        name = match["slashed"] or ""
    return _Node(role=match["role"], name=name, value=value, attrs=match["attrs"])


def parse_snapshot(snapshot: str) -> _Node:
    """Parse an ARIA snapshot into a tree; raise SnapshotFormatError on surprises."""
    root = _Node(role="root")
    stack: list[tuple[int, _Node]] = [(-1, root)]
    for line in snapshot.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if not body.startswith("- "):
            raise SnapshotFormatError(line)
        while stack[-1][0] >= indent:
            stack.pop()
        node = _parse_entry(body[2:])
        stack[-1][1].children.append(node)
        stack.append((indent, node))
    return root


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _own_strings(node: _Node) -> list[str]:
    """The node's own visible strings: name and value, minus link targets and glyphs."""
    if node.role == "/url":
        return []
    strings = [text for text in (node.name, node.value) if text and not _NON_TEXT_ONLY_RE.match(text)]
    if len(strings) == 2 and _collapse(strings[0]) == _collapse(strings[1]):
        strings.pop()
    return strings


def _all_strings(node: _Node) -> list[str]:
    strings = _own_strings(node)
    for child in node.children:
        strings.extend(_all_strings(child))
    return strings


def _children_repeat_name(node: _Node) -> bool:
    if not node.name:
        return False
    name = _collapse(node.name)
    return all(_collapse(text) in name for child in node.children for text in _all_strings(child))


def _inline_text(node: _Node) -> str:
    """All text of a subtree on one line, with repeated names left out."""
    parts = _own_strings(node)
    if not _children_repeat_name(node):
        parts.extend(text for child in node.children if (text := _inline_text(child)))
    return " ".join(_collapse(part) for part in parts)


def _render(node: _Node, lines: list[str]) -> None:
    strings = _own_strings(node)
    if node.role == "row" and node.children and all(child.role in _CELL_ROLES for child in node.children):
        # A row's name repeats its cells; render the cells so the table keeps its columns.
        cells = [_inline_text(child) for child in node.children]
        row = "| " + " | ".join(cells) + " |"
        if any(_collapse(text) not in row for text in strings):
            lines.extend(strings)
        lines.append(row)
        return
    children = [] if _children_repeat_name(node) else node.children
    if strings:
        line = " ".join(strings)
        if node.role == "heading":
            level = _LEVEL_RE.search(node.attrs)
            line = "#" * int(level.group(1) if level else 2) + " " + line
        elif node.role == "listitem":
            line = "- " + line
        lines.append(line)
    elif node.role == "listitem" and children:
        start = len(lines)
        for child in children:
            _render(child, lines)
        if len(lines) > start:
            lines[start] = "- " + lines[start]
        return
    for child in children:
        _render(child, lines)


def clean_snapshot(snapshot: str) -> str:
    """Plain reading text for an ARIA snapshot; the snapshot itself if it won't parse."""
    try:
        tree = parse_snapshot(snapshot)
    except (SnapshotFormatError, ValueError):
        return snapshot
    rendered: list[str] = []
    _render(tree, rendered)
    lines: list[str] = []
    for line in rendered:
        line = line.strip()
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return "\n".join(lines) if lines else snapshot
