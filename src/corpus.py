"""The FastAPI docs as a reader of the website sees them.

The markdown files under docs/en/docs are the *input* to a docs build, not the
documentation. Most of the code a reader sees is not in them: a page says

    {* ../../docs_src/query_params/tutorial002_py310.py hl[7] *}

and the build pastes that file in. Index the raw files and the answer to "how
do I make a query parameter optional?" is a page whose example has been
replaced by a file path.

So this module does the part of the build that changes which words are on the
page, and nothing else:

* code includes are resolved -- `{* ... *}` and `{! ... !}` -- with `ln[...]`
  line ranges and the site's "Code above omitted" markers. The collapsed
  "Full file preview" the site adds under an excerpt is left out: it repeats
  the file, and near-duplicate chunks would crowd each other out of a top-k;
* admonition and tab markers (`/// tip`, `//// tab | Python 3.10+`) become a
  one-line label;
* heading anchors (`{ #defaults }`) are taken out of the text and kept as
  metadata, so any character offset can be cited as a section URL;
* HTML becomes its text -- except inside code blocks, where HTML is the code;
* front matter and the sponsor-banner template loops are dropped.

Markdown emphasis, lists and tables stay: that is text a reader sees too.
"""

from __future__ import annotations

import bisect
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import config

# Pages in docs/en/docs that do not document how to use FastAPI. Matched as a
# prefix of the path relative to docs/en/docs. scripts/01_corpus_stats.py
# prints each reason, so the corpus can be audited without reading this file.
EXCLUDED: dict[str, str] = {
    "release-notes.md": "changelog, not documentation -- and it alone would be almost a third of the index",
    "reference/": "API reference, generated from docstrings at build time; the pages are `::: fastapi.X` stubs",
    "fastapi-people.md": "generated at build time from contributor data",
    "external-links.md": "generated at build time from a list of links",
    "_llm-test.md": "test fixture for the docs' translation tooling",
    "translation-banner.md": "banner shown on translated pages",
    "translations.md": "about the project: contributing translations",
    "management.md": "about the project: repository management",
    "contributing.md": "about the project: contributing code",
    "help-fastapi.md": "about the project: helping it, and where to ask",
    "newsletter.md": "about the project: newsletter sign-up",
}


def exclusion_reason(doc_id: str) -> str | None:
    for prefix, reason in EXCLUDED.items():
        if doc_id == prefix or (prefix.endswith("/") and doc_id.startswith(prefix)):
            return reason
    return None


class IncludeError(ValueError):
    """An include directive that cannot be resolved.

    Raised, never skipped: a page that silently loses its code example still
    renders and still gets indexed, and nothing downstream would notice.
    """


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    anchor: str
    offset: int  # where the heading line starts in Document.text


@dataclass(frozen=True)
class Document:
    doc_id: str  # path under docs/en/docs, e.g. "tutorial/query-params.md"
    url: str
    title: str
    text: str
    headings: tuple[Heading, ...]
    includes: int = 0  # code includes resolved into this page
    included_chars: int = 0  # characters of `text` that came from them

    def section_at(self, offset: int) -> Heading | None:
        """The most specific section that `offset` falls in.

        In a linear document the last heading at or before a position is the
        deepest section still open there, whatever its level.
        """
        i = bisect.bisect_right([h.offset for h in self.headings], offset) - 1
        return self.headings[i] if i >= 0 else None

    def cite(self, offset: int) -> str:
        """URL of the section containing `offset` -- what an answer links to."""
        heading = self.section_at(offset)
        if heading is None or heading.level == 1:
            return self.url
        return f"{self.url}#{heading.anchor}"


def page_url(doc_id: str) -> str:
    """The published URL of a page (mkdocs directory URLs)."""
    stem = doc_id.removesuffix(".md")
    if stem == "index":
        return config.SITE_URL
    stem = stem.removesuffix("/index")
    return f"{config.SITE_URL}{stem}/"


# ---------------------------------------------------------------- patterns

_FENCE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_CODE_INCLUDE = re.compile(r"^\s*\{\*\s*(?P<path>\S+)(?P<opts>.*?)\*\}\s*$")
_MDX_INCLUDE = re.compile(r"^\s*\{!>?\s*(?P<path>[^!]+?)\s*!\}\s*$")
_LINE_RANGES = re.compile(r"\bln\[(?P<spec>[^\]]*)\]")
# Indented when the block sits inside a list item; one page capitalises "Note".
_BLOCK_OPEN = re.compile(r"^\s*/{3,4}\s*(?P<kind>[A-Za-z]+)\s*(?:\|\s*(?P<title>.*?))?\s*$")
_BLOCK_CLOSE = re.compile(r"^\s*/{3,4}\s*$")
_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.*?)\s*(?:\{\s*#(?P<anchor>[^\s}]+)\s*\})?\s*$")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_TEMPLATE_LOOP = re.compile(r"\{%-?\s*for\b.*?\{%-?\s*endfor\s*-?%\}", re.S)
_TEMPLATE_RAW = re.compile(r"^\s*\{%-?\s*(?:end)?raw\s*-?%\}\s*$")

_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_DEFINITION = re.compile(
    r"<(?P<tag>dfn|abbr)\s+title=(?P<q>[\"'])(?P<title>.*?)(?P=q)\s*>(?P<text>.*?)</(?P=tag)>",
    re.I,
)
_DROPPED_TAG = re.compile(r"<(?:img|iframe)\b[^>]*>(?:\s*</iframe>)?", re.I)
_AUTOLINK = re.compile(r"<(https?://[^<>\s]+)>")
_ANY_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sh": "bash",
}


# ---------------------------------------------------------------- pieces


def _prose(line: str) -> str:
    """Reduce one line of prose to the text a reader sees.

    Inline code spans are set aside first and put back last: `<a>` written in
    backticks is something the page is *talking about*, not markup.
    """
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    s = _INLINE_CODE.sub(stash, line)
    # A definition's tooltip is text the reader can see on hover, and it is
    # often the only plain-English gloss of a term -- keep it inline.
    s = _DEFINITION.sub(lambda m: f"{m.group('text')} ({m.group('title')})", s)
    s = _DROPPED_TAG.sub("", s)
    s = _IMAGE.sub("", s)
    s = _LINK.sub(lambda m: m.group("text"), s)
    s = _AUTOLINK.sub(r"\1", s)  # `<https://...>` is a link, not a tag
    s = _ANY_TAG.sub("", s)
    s = html.unescape(s)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], s)


def _slugify(title: str) -> str:
    """Python-Markdown's default heading id, for the few headings without one."""
    s = re.sub(r"[`*_]", "", title)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\s]+", "-", s)


# The markers the docs site prints around an excerpt, copied verbatim: they
# tell the reader -- and the model -- that the snippet is not the whole file.
_OMITTED_ABOVE = "# Code above omitted 👆"
_OMITTED_HERE = "# Code here omitted 👈"
_OMITTED_BELOW = "# Code below omitted 👇"


def _select_lines(lines: list[str], spec: str) -> list[str]:
    """Apply an `ln[1:9,29:35]` spec: 1-based, inclusive, comma-separated."""
    picked: list[str] = []
    end = 0
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        start_s, _, end_s = part.partition(":") if ":" in part else (part, "", part)
        start = int(start_s) if start_s else 1
        end = int(end_s) if end_s else len(lines)
        if not 1 <= start <= end <= len(lines):
            raise IncludeError(f"line range {part!r} outside a {len(lines)}-line file")
        if picked:
            picked += ["", _OMITTED_HERE, ""]
        elif start > 1:
            picked += [_OMITTED_ABOVE, ""]
        picked.extend(lines[start - 1 : end])
    if picked and end < len(lines):
        picked += ["", _OMITTED_BELOW]
    return picked


class _Includes:
    """Resolves include paths the way the docs build does, and keeps count."""

    def __init__(self, build_dir: Path, root: Path, doc_id: str) -> None:
        self.build_dir = build_dir
        self.root = root.resolve()
        self.doc_id = doc_id
        self.count = 0
        self.chars = 0

    def read(self, rel_path: str) -> tuple[Path, list[str]]:
        path = (self.build_dir / rel_path).resolve()
        if not path.is_relative_to(self.root):
            raise IncludeError(f"{self.doc_id}: include escapes the corpus: {rel_path}")
        if not path.is_file():
            raise IncludeError(f"{self.doc_id}: include not found: {rel_path}")
        return path, path.read_text(encoding="utf-8").rstrip("\n").split("\n")

    def lines(self, rel_path: str, opts: str = "") -> tuple[str, list[str]]:
        """(language, lines) for one directive."""
        path, lines = self.read(rel_path)
        language = _LANGUAGE.get(path.suffix, "")
        ranges = _LINE_RANGES.search(opts)
        if ranges:
            try:
                lines = _select_lines(lines, ranges.group("spec"))
            except IncludeError as exc:
                raise IncludeError(f"{self.doc_id}: {rel_path}: {exc}") from None
        self.count += 1
        self.chars += sum(len(line) + 1 for line in lines)
        return language, lines


# ---------------------------------------------------------------- render


def render(
    doc_id: str,
    markdown: str,
    build_dir: Path | None = None,
    root: Path | None = None,
) -> Document:
    """Render one page. `build_dir` is where include paths are relative to."""
    build_dir = build_dir or config.DOCS_BUILD_DIR
    includes = _Includes(build_dir, root or config.CORPUS_DIR, doc_id)

    text = markdown.replace("\r\n", "\n")
    text = _FRONT_MATTER.sub("", text)
    text = _TEMPLATE_LOOP.sub("", text)

    out: list[str] = []
    pos = 0  # length of "\n".join(out) + "\n", i.e. where the next line starts
    headings: list[Heading] = []
    seen_anchors: set[str] = set()

    def emit(line: str, in_code: bool = False) -> None:
        nonlocal pos
        # Outside code, runs of blank lines collapse to one; inside code the
        # blank lines are part of the example.
        if not in_code and not line.strip() and (not out or not out[-1].strip()):
            return
        out.append(line)
        pos += len(line) + 1

    fence: str | None = None  # the opening fence while inside a code block
    fence_info = ""

    for line in text.split("\n"):
        if fence is not None:
            m = _FENCE.match(line)
            if m and m.group("fence")[0] == fence[0] and len(m.group("fence")) >= len(fence) and not m.group("info").strip():
                fence = None
                emit(line, in_code=True)
                continue
            inc = _MDX_INCLUDE.match(line) or _CODE_INCLUDE.match(line)
            if inc:
                _, lines = includes.lines(inc.group("path"), inc.groupdict().get("opts") or "")
                for code_line in lines:
                    emit(code_line, in_code=True)
                continue
            if fence_info.startswith("console"):
                # Terminal colours, written as HTML for the docs' terminal
                # widget. The reader sees the text, not the tags.
                line = html.unescape(_ANY_TAG.sub("", line))
            emit(line, in_code=True)
            continue

        m = _FENCE.match(line)
        if m:
            fence = m.group("fence")
            fence_info = m.group("info").strip().lower()
            emit(line, in_code=True)
            continue

        if _TEMPLATE_RAW.match(line):
            continue

        inc = _CODE_INCLUDE.match(line)
        if inc:
            language, lines = includes.lines(inc.group("path"), inc.group("opts"))
            emit(f"```{language}", in_code=True)
            for code_line in lines:
                emit(code_line, in_code=True)
            emit("```", in_code=True)
            continue

        inc = _MDX_INCLUDE.match(line)
        if inc:
            _, lines = includes.lines(inc.group("path"))
            for code_line in lines:
                emit(code_line, in_code=True)
            continue

        if _BLOCK_CLOSE.match(line):
            emit("")
            continue
        m = _BLOCK_OPEN.match(line)
        if m:
            label = m.group("title") or m.group("kind").capitalize()
            emit(f"**{_prose(label).strip()}**")
            continue

        m = _HEADING.match(line)
        if m:
            title = _prose(m.group("title")).strip()
            anchor = m.group("anchor") or _slugify(title)
            if not m.group("anchor"):
                # Python-Markdown de-duplicates generated ids the same way.
                base, n = anchor, 1
                while anchor in seen_anchors:
                    anchor, n = f"{base}_{n}", n + 1
            seen_anchors.add(anchor)
            level = len(m.group("hashes"))
            headings.append(Heading(level, title, anchor, pos))
            emit(f"{m.group('hashes')} {title}")
            continue

        emit(_prose(line).rstrip())

    if fence is not None:
        raise ValueError(f"{doc_id}: unclosed code fence {fence!r}")

    h1 = next((h for h in headings if h.level == 1), None)
    return Document(
        doc_id=doc_id,
        url=page_url(doc_id),
        title=h1.title if h1 else doc_id,
        text="\n".join(out).rstrip(),
        headings=tuple(headings),
        includes=includes.count,
        included_chars=includes.chars,
    )


# ---------------------------------------------------------------- loading


def page_ids(docs_dir: Path | None = None) -> list[str]:
    """Every English page, as a path relative to docs/en/docs, sorted."""
    docs_dir = docs_dir or config.DOCS_DIR
    return sorted(p.relative_to(docs_dir).as_posix() for p in docs_dir.rglob("*.md"))


def load_page(doc_id: str, docs_dir: Path | None = None) -> Document:
    docs_dir = docs_dir or config.DOCS_DIR
    return render(doc_id, (docs_dir / doc_id).read_text(encoding="utf-8"))


def load_corpus() -> list[Document]:
    """Every page that documents how to use FastAPI, rendered, sorted by path."""
    config.require_data()
    return [load_page(d) for d in page_ids() if exclusion_reason(d) is None]
