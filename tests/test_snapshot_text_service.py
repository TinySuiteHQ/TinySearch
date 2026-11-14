from __future__ import annotations

import re
import unittest

from tinysearch.services.snapshot_text_service import (
    _NON_TEXT_ONLY_RE,
    SnapshotFormatError,
    clean_snapshot,
    parse_snapshot,
)

try:  # Optional: an independent parser for the no-text-lost check.
    import yaml
except ImportError:  # pragma: no cover - depends on the environment
    yaml = None


# Every line shape Playwright's AI-mode snapshots were seen to use, with page
# text in several scripts and directions. The text is made up for the test.
MULTILINGUAL_SNAPSHOT = """
- banner:
  - navigation:
    - list:
      - listitem:
        - link "Startseite":
          - /url: /
      - listitem:
        - link "Über uns":
          - /url: /about
    - button ""
    - text: 
- main:
  - heading "布达佩斯的历史" [level=1]
  - paragraph: 布达佩斯是匈牙利的首都，人口约一百七十万。
  - heading "Будапешт" [level=2]
  - paragraph: "Город на Дунае: две части, Буда и Пешт."
  - link "مدينة على نهر الدانوب":
    - /url: https://example.com/ar
    - generic: مدينة على نهر الدانوب
  - paragraph: עיר הבירה של הונגריה
  - 'link "Tóta W.: „Ideje széttépni”"':
    - /url: /hu
  - link /ˈbuːdəpɛst/:
    - /url: "#ipa"
  - paragraph: बुडापेस्ट हंगरी की राजधानी है।
  - text: ช้า‌ง ப‌ண்டை میخواهم
  - img "Ο ποταμός Δούναβης"
  - img
  - checkbox "日本語のニュース" [checked]
  - table:
    - rowgroup:
      - row "Év Népesség":
        - columnheader "Év"
        - columnheader "Népesség"
      - row "2024 1 685 342":
        - cell "2024"
        - cell "1 685 342"
  - list:
    - listitem:
      - generic:
        - generic: 서울과 부다페스트
    - listitem: Tiếng Việt có dấu
  - textbox "Keresés":
    - /placeholder: Mit keresel?
  - "text: a colon: inside a quoted item"
- contentinfo:
  - paragraph: © 2026 Example
""".strip("\n")


def _visible_strings(snapshot: str) -> list[str]:
    """Every visible string in a snapshot, read with PyYAML instead of our parser."""
    name_re = re.compile(r'^[a-z][\w-]*(?: "((?:[^"\\]|\\.)*)"| (/.*?/))?')
    strings: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for key, child in value.items():
                key = str(key)
                if key == "/url":
                    continue
                match = name_re.match(key)
                if match and match.group(1):
                    strings.append(yaml.safe_load('"' + match.group(1) + '"'))
                elif match and match.group(2):
                    strings.append(match.group(2))
                elif not match and not key.startswith("/"):
                    strings.append(key)
                walk(child)
        elif isinstance(value, str):
            match = name_re.match(value)
            bare_role = re.fullmatch(r"[a-z][\w-]*(?: \[[^\]]*\])*", value)
            if match and match.group(1):
                strings.append(yaml.safe_load('"' + match.group(1) + '"'))
            elif match and match.group(2):
                strings.append(match.group(2))
            elif not bare_role:
                strings.append(value)

    walk(yaml.load(snapshot, Loader=yaml.BaseLoader))
    return [text for text in strings if text and not _NON_TEXT_ONLY_RE.match(text)]


def _collapse(text: str) -> str:
    return " ".join(text.split())


class CleanSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cleaned = clean_snapshot(MULTILINGUAL_SNAPSHOT)
        self.lines = self.cleaned.splitlines()

    def test_drops_link_targets_role_keywords_and_attributes(self) -> None:
        for leftover in ("/url", "https://example.com/ar", "#ipa", "- link", "generic:", "[level=", "[checked]"):
            self.assertNotIn(leftover, self.cleaned)

    def test_keeps_text_in_every_script(self) -> None:
        for text in (
            "布达佩斯是匈牙利的首都，人口约一百七十万。",
            "Город на Дунае: две части, Буда и Пешт.",
            "مدينة على نهر الدانوب",
            "עיר הבירה של הונגריה",
            "बुडापेस्ट हंगरी की राजधानी है।",
            "Ο ποταμός Δούναβης",
            "日本語のニュース",
            "서울과 부다페스트",
            "Tiếng Việt có dấu",
        ):
            self.assertIn(text, self.cleaned)

    def test_keeps_zero_width_joiners_inside_text(self) -> None:
        self.assertIn("ช้า‌ง ப‌ண்டை میخواهم", self.cleaned)

    def test_drops_strings_made_only_of_icon_glyphs(self) -> None:
        self.assertNotIn("", self.cleaned)
        self.assertNotIn("", self.cleaned)

    def test_marks_headings_list_items_and_table_rows(self) -> None:
        self.assertIn("# 布达佩斯的历史", self.lines)
        self.assertIn("## Будапешт", self.lines)
        self.assertIn("- Startseite", self.lines)
        self.assertIn("- 서울과 부다페스트", self.lines)
        self.assertIn("- Tiếng Việt có dấu", self.lines)
        self.assertIn("| Év | Népesség |", self.lines)
        self.assertIn("| 2024 | 1 685 342 |", self.lines)

    def test_children_that_repeat_the_name_appear_once(self) -> None:
        self.assertEqual(self.cleaned.count("مدينة على نهر الدانوب"), 1)

    def test_children_with_new_text_are_kept(self) -> None:
        cleaned = clean_snapshot('- link "Read more":\n  - generic: Full story about the bridge')
        self.assertEqual(cleaned, "Read more\nFull story about the bridge")

    def test_reads_quoted_and_slashed_names_and_quoted_values(self) -> None:
        self.assertIn("Tóta W.: „Ideje széttépni”", self.lines)
        self.assertIn("/ˈbuːdəpɛst/", self.lines)
        self.assertIn("text: a colon: inside a quoted item", self.lines)

    def test_keeps_labels_of_controls_and_images(self) -> None:
        self.assertIn("Keresés", self.lines)
        self.assertIn("Mit keresel?", self.lines)

    @unittest.skipIf(yaml is None, "PyYAML is not installed")
    def test_no_visible_string_is_lost(self) -> None:
        haystack = _collapse(self.cleaned)
        missing = [text for text in _visible_strings(MULTILINGUAL_SNAPSHOT) if _collapse(text) not in haystack]
        self.assertEqual(missing, [])

    def test_collapses_consecutive_identical_lines(self) -> None:
        self.assertEqual(clean_snapshot("- text: Same\n- text: Same\n- text: Other"), "Same\nOther")

    def test_unrecognised_shapes_return_the_snapshot_unchanged(self) -> None:
        for snapshot in (
            "- heading \"Unclosed [level=1]\n- text: fine",
            "plain text that is not an outline",
            "- 'quoted header' trailing junk",
        ):
            self.assertEqual(clean_snapshot(snapshot), snapshot)

    def test_snapshot_without_text_is_returned_unchanged(self) -> None:
        self.assertEqual(clean_snapshot("- generic:\n  - img"), "- generic:\n  - img")

    def test_parser_raises_on_surprises(self) -> None:
        with self.assertRaises(SnapshotFormatError):
            parse_snapshot("no dash here")


if __name__ == "__main__":
    unittest.main()
