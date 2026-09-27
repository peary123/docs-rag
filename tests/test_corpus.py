"""Tests for rendering the docs into the text that gets indexed.

The failure this guards against is quiet: a page that loses its code example,
or keeps `/// tip` and `{ #anchor }` noise, still renders and still gets
indexed, and retrieval is just a little worse for reasons nobody can see.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, corpus  # noqa: E402

HAVE_DATA = config.DOCS_DIR.exists()

APP_PY = "\n".join(f"x{i} = {i}" for i in range(1, 11)) + "\n"
ITEM_HTML = '<a href="{{ url_for(\'item\', id=id) }}">Item</a>\n'


def _render(markdown: str) -> corpus.Document:
    """Render against a throwaway corpus laid out like the real one."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs" / "en").mkdir(parents=True)
        (root / "docs_src" / "demo").mkdir(parents=True)
        (root / "docs_src" / "demo" / "app.py").write_text(APP_PY, encoding="utf-8")
        (root / "docs_src" / "demo" / "item.html").write_text(ITEM_HTML, encoding="utf-8")
        return corpus.render("page.md", markdown, build_dir=root / "docs" / "en", root=root)


def _raises(markdown: str, fragment: str) -> None:
    try:
        _render(markdown)
    except corpus.IncludeError as exc:
        assert fragment in str(exc), exc
        return
    raise AssertionError("expected IncludeError")


# ------------------------------------------------------------------ includes


def test_code_include_is_pasted_in_as_a_fenced_block() -> None:
    doc = _render("# T { #t }\n\n{* ../../docs_src/demo/app.py hl[3] *}\n\nAfter.")
    assert "```python\nx1 = 1\n" in doc.text
    assert "x10 = 10\n```" in doc.text
    assert "{*" not in doc.text
    assert doc.includes == 1 and doc.included_chars == len(APP_PY)


def test_line_ranges_are_one_based_inclusive_with_the_sites_markers() -> None:
    doc = _render("{* ../../docs_src/demo/app.py ln[2:3,6:7] hl[2] *}")
    code = doc.text.split("\n")[1:-1]  # inside the fence
    assert code == [
        "# Code above omitted 👆",
        "",
        "x2 = 2",
        "x3 = 3",
        "",
        "# Code here omitted 👈",
        "",
        "x6 = 6",
        "x7 = 7",
        "",
        "# Code below omitted 👇",
    ]


def test_a_range_that_reaches_the_end_has_no_below_marker() -> None:
    doc = _render("{* ../../docs_src/demo/app.py ln[9:10] *}")
    assert "below omitted" not in doc.text
    assert doc.text.rstrip("`\n").endswith("x10 = 10")


def test_include_inside_a_fence_is_pasted_raw() -> None:
    doc = _render("```jinja\n{!../../docs_src/demo/item.html!}\n```")
    assert doc.text == "```jinja\n" + ITEM_HTML.rstrip("\n") + "\n```"


def test_missing_include_raises() -> None:
    _raises("{* ../../docs_src/demo/nope.py *}", "include not found")


def test_include_cannot_escape_the_corpus() -> None:
    _raises("{* ../../../../etc/passwd *}", "escapes the corpus")


def test_line_range_past_the_end_raises() -> None:
    """How a stale `ln[...]` after an upstream edit would show up."""
    _raises("{* ../../docs_src/demo/app.py ln[8:12] *}", "outside a 10-line file")


# ------------------------------------------------------------------ structure


def test_heading_anchor_is_metadata_not_text() -> None:
    doc = _render("# Query { #query }\n\nIntro.\n\n## Defaults { #defaults }\n\nBody text.")
    assert "{ #" not in doc.text
    defaults = doc.headings[1]
    assert (defaults.level, defaults.title, defaults.anchor) == (2, "Defaults", "defaults")
    assert doc.text[defaults.offset :].startswith("## Defaults\n")
    assert doc.cite(doc.text.index("Body text")) == doc.url + "#defaults"
    # The h1 is the page itself, so its section cites the bare page URL.
    assert doc.cite(doc.text.index("Intro")) == doc.url
    assert doc.title == "Query"


def test_python_comment_in_a_code_block_is_not_a_heading() -> None:
    doc = _render("# Title { #title }\n\n```python\n# This is not a heading\nx = 1\n```")
    assert [h.title for h in doc.headings] == ["Title"]


def test_heading_without_an_anchor_gets_the_sites_slug() -> None:
    doc = _render("# `Depends` and You\n\n## Same\n\n## Same")
    assert [h.anchor for h in doc.headings] == ["depends-and-you", "same", "same_1"]


def test_admonitions_and_tabs_become_labels() -> None:
    doc = _render(
        "/// tip\n\nUse it.\n\n///\n\n"
        "//// tab | Python 3.10+\n\ncode\n\n////\n\n"
        "1. Item\n\n    /// note\n\n    Indented, inside a list.\n\n    ///\n\n"
        "/// Note\n\nCapitalised.\n\n///\n\n"
        "/// note | Technical Details\n\nDetails.\n\n///"
    )
    assert "**Tip**" in doc.text
    assert "**Python 3.10+**" in doc.text
    assert doc.text.count("**Note**") == 2
    assert "**Technical Details**" in doc.text
    assert not any(line.lstrip().startswith("///") for line in doc.text.split("\n"))


def test_runs_of_blank_lines_collapse_outside_code_only() -> None:
    doc = _render("A\n\n\n\nB\n\n```python\nx = 1\n\n\ny = 2\n```")
    assert "A\n\nB" in doc.text
    assert "x = 1\n\n\ny = 2" in doc.text


# ------------------------------------------------------------------ HTML


def test_html_becomes_its_text_outside_code() -> None:
    doc = _render(
        "The <dfn title='also called \"bitwise or\"'>vertical bar (`|`)</dfn> joins types. "
        '<img src="x.png"> See [the tutorial](../tutorial/index.md) or <https://example.com>. '
        "Use &lt;T&gt;."
    )
    assert 'vertical bar (`|`) (also called "bitwise or") joins types.' in doc.text
    assert "img" not in doc.text and "x.png" not in doc.text
    assert "See the tutorial or https://example.com." in doc.text
    assert "Use <T>." in doc.text


def test_html_in_backticks_or_in_an_html_example_is_kept() -> None:
    doc = _render('Return an `<a>` tag:\n\n```html\n<a href="{{ url }}">x</a>\n```')
    assert "Return an `<a>` tag:" in doc.text
    assert '<a href="{{ url }}">x</a>' in doc.text


def test_terminal_colour_markup_is_stripped_from_console_blocks() -> None:
    doc = _render('```console\n$ <font color="#4E9A06">uv run fastapi</font> dev\n```')
    assert "$ uv run fastapi dev" in doc.text


def test_front_matter_sponsor_loop_and_raw_markers_are_dropped() -> None:
    doc = _render(
        "---\ninclude_yaml:\n  sponsors: data/sponsors.yml\n---\n"
        "# Home { #home }\n\n"
        "{% for sponsor in sponsors.gold -%}\n"
        '<a href="{{ sponsor.url }}"><img src="{{ sponsor.img }}"></a>\n'
        "{% endfor -%}\n\n"
        "{% raw %}\n\n```jinja\nItem ID: {{ id }}\n```\n\n{% endraw %}\n"
    )
    assert doc.text.startswith("# Home")
    assert "sponsor" not in doc.text and "{%" not in doc.text
    assert "Item ID: {{ id }}" in doc.text


# ------------------------------------------------------------------ pages


def test_page_urls_follow_the_site() -> None:
    site = config.SITE_URL
    assert corpus.page_url("index.md") == site
    assert corpus.page_url("tutorial/index.md") == site + "tutorial/"
    assert corpus.page_url("tutorial/query-params.md") == site + "tutorial/query-params/"


def test_exclusions_match_whole_names_or_directories() -> None:
    assert corpus.exclusion_reason("release-notes.md")
    assert corpus.exclusion_reason("reference/openapi/docs.md")
    assert corpus.exclusion_reason("tutorial/index.md") is None
    assert corpus.exclusion_reason("referencing.md") is None


# ------------------------------------------------------------------ the corpus


def test_every_page_renders_and_every_include_resolves() -> None:
    """All 155 pages, excluded ones too: a stale path fails here, loudly."""
    if not HAVE_DATA:
        return
    pages = [corpus.load_page(d) for d in corpus.page_ids()]
    assert len(pages) == 155
    assert sum(p.includes for p in pages) == 446


def test_the_indexed_pages_are_clean() -> None:
    if not HAVE_DATA:
        return
    docs = corpus.load_corpus()
    assert len(docs) == 121
    assert len({d.url for d in docs}) == len(docs)
    for d in docs:
        assert any(h.level == 1 for h in d.headings), d.doc_id
        anchors = [h.anchor for h in d.headings]
        assert len(anchors) == len(set(anchors)), d.doc_id
        for h in d.headings:
            assert d.text[h.offset :].startswith("#" * h.level + " " + h.title), (d.doc_id, h)
        for line in d.text.split("\n"):
            stripped = line.lstrip()
            assert not stripped.startswith(("///", "{*", "{!", "{%")), (d.doc_id, line)
            assert "{ #" not in line, (d.doc_id, line)


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {name}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
