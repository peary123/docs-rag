"""The question set: sampling sections, locating evidence, and checking it.

A question's gold answer location is stored as **character spans in the
rendered page text**, not as chunk ids. Retrieval is compared across two
chunking strategies, so an id from one chunking does not exist in the other.
A span works for both: a retrieved chunk counts as a hit when it overlaps one.

Every span also keeps the exact text it covers. If the corpus renderer ever
changes, `validate` notices that the stored quote no longer matches the text at
those offsets, instead of the question set silently pointing at the wrong words.
"""

from __future__ import annotations

import bisect
import functools
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .corpus import Document

# Sections shorter than this rarely hold a question worth asking: they are
# landing-page blurbs, one-line notes, or a heading over a single code block.
MIN_SECTION_CHARS = 400

# Sections the question set must not be drawn from, each with the reason.
EXCLUDED_SECTIONS = {
    "how-to/configure-swagger-ui.md#change-default-swagger-ui-parameters": (
        "the page includes lines of fastapi/openapi/docs.py that no longer hold "
        "the defaults its text describes (NOTES.md: 'The docs show the wrong "
        "code on one page'), so any answer drawn from it would be wrong"
    ),
}


# ------------------------------------------------------------------ sections

@dataclass(frozen=True)
class Section:
    """The text from one heading to the next heading of any level."""

    doc_id: str
    index: int  # position among this page's sections
    anchor: str
    title: str
    level: int
    start: int  # offset of the heading line in Document.text
    end: int  # offset of the next heading, or the end of the text

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.anchor}"

    def text(self, doc: Document) -> str:
        return doc.text[self.start:self.end]


def sections(doc: Document) -> list[Section]:
    ends = [h.offset for h in doc.headings[1:]] + [len(doc.text)]
    return [
        Section(doc.doc_id, i, h.anchor, h.title, h.level, h.offset, end)
        for i, (h, end) in enumerate(zip(doc.headings, ends))
    ]


def eligible_sections(docs: list[Document]) -> dict[str, list[Section]]:
    """Sections a candidate question may be drawn from, grouped by page."""
    out: dict[str, list[Section]] = {}
    for doc in docs:
        keep = [
            s for s in sections(doc)
            if s.end - s.start >= MIN_SECTION_CHARS and s.key not in EXCLUDED_SECTIONS
        ]
        if keep:
            out[doc.doc_id] = keep
    return out


def sample_sections(docs: list[Document], n: int, seed: int) -> list[Section]:
    """`n` sections spread across pages: one per page before any page gets two.

    Drawing uniformly from all sections would favour the long tutorial pages;
    the question set should cover the docs, not the docs' longest pages.
    Deterministic for a given seed, so the candidate set can be regenerated.
    """
    rng = random.Random(seed)
    by_page = eligible_sections(docs)
    pages = sorted(by_page)
    rng.shuffle(pages)
    pools = {p: rng.sample(by_page[p], len(by_page[p])) for p in pages}
    picked: list[Section] = []
    while len(picked) < n and any(pools.values()):
        for page in pages:
            if pools[page] and len(picked) < n:
                picked.append(pools[page].pop())
    return picked


def sample_section_pairs(docs: list[Document], n: int, seed: int,
                         avoid: set[str] = frozenset()) -> list[tuple[Section, Section]]:
    """`n` pairs of nearby sections on the same page, each page used once.

    Nearby (one to three sections apart) because unrelated sections rarely make
    a natural question that needs both. Whether a pair really needs both halves
    is for the reviewer to judge; this only offers plausible pairs.
    """
    rng = random.Random(seed)
    by_page = {p: [s for s in ss if s.key not in avoid]
               for p, ss in eligible_sections(docs).items()}
    pages = sorted(p for p, ss in by_page.items() if len(ss) >= 2)
    rng.shuffle(pages)
    pairs: list[tuple[Section, Section]] = []
    for page in pages:
        ss = by_page[page]
        options = [(a, b) for i, a in enumerate(ss) for b in ss[i + 1:]
                   if 1 <= b.index - a.index <= 3]
        if options:
            pairs.append(rng.choice(options))
        if len(pairs) == n:
            break
    return pairs


# ------------------------------------------------------------------ evidence

@dataclass
class Evidence:
    doc_id: str
    start: int
    end: int
    anchor: str  # section the span starts in -- what an answer would cite
    quote: str  # the exact text at [start:end], kept to detect drift

    def as_dict(self) -> dict:
        return asdict(self)


# Characters that change how text looks but not what it says. Quote marks are
# on the list because a model writing JSON swaps `"` for `'` inside strings to
# avoid escaping them -- the first generation run lost most of its rejected
# candidates to exactly that, with every word intact.
_FORMATTING = set("\"'`‘’“”*")
_ELISION = re.compile(r"\.\.\.|…")


@functools.lru_cache(maxsize=256)
def _normalised(s: str) -> tuple[str, tuple[int, ...]]:
    """`s` with formatting characters dropped and whitespace runs collapsed to
    one space, plus the index in `s` that each remaining character came from."""
    out: list[str] = []
    idx: list[int] = []
    in_space = False
    for i, ch in enumerate(s):
        if ch in _FORMATTING:
            continue
        if ch.isspace():
            if not in_space:
                out.append(" ")
                idx.append(i)
            in_space = True
            continue
        in_space = False
        out.append(ch)
        idx.append(i)
    return "".join(out), tuple(idx)


def locate(text: str, quote: str, start: int = 0, end: int | None = None) -> tuple[int, int] | None:
    """Find `quote` in text[start:end]: the same words, in the same order.

    Whitespace, quote-mark style, backticks and `*` emphasis are ignored; the
    words are not. A model asked to copy a passage reflows code indentation,
    drops bold markers and swaps quote marks, but a changed word is a paraphrase
    and does not match -- the candidate is rejected rather than guessed at.

    A quote that skips text with "..." matches when every piece is found in
    order; the span then runs from the first piece to the last.
    """
    stop = len(text) if end is None else end
    stripped = quote.strip()
    if not stripped:
        return None
    exact = text.find(stripped, start, stop)
    if exact >= 0:
        return exact, exact + len(stripped)

    pieces = [p for p in (_normalised(part)[0].strip() for part in _ELISION.split(stripped)) if p]
    if not pieces:
        return None
    norm, idx = _normalised(text)
    lo, hi = bisect.bisect_left(idx, start), bisect.bisect_left(idx, stop)
    first = last = None
    for piece in pieces:
        found = norm.find(piece, lo, hi)
        if found < 0:
            return None
        first = found if first is None else first
        last = lo = found + len(piece)
    return idx[first], idx[last - 1] + 1


def make_evidence(doc: Document, start: int, end: int) -> Evidence:
    heading = doc.section_at(start)
    return Evidence(doc.doc_id, start, end, heading.anchor if heading else "", doc.text[start:end])


# ------------------------------------------------------------------ overlap

_WORD = re.compile(r"[A-Za-z0-9_]+")


def longest_shared_run(question: str, passage: str) -> int:
    """Longest run of consecutive question words that also occurs in the passage.

    Questions a model writes from a passage tend to reuse its wording, and a
    question set full of them rewards keyword search (BM25) for reasons that
    have nothing to do with how real users ask. This is shown to the reviewer as
    a warning, and reported for the final set.
    """
    q = [w.lower() for w in _WORD.findall(question)]
    p = [w.lower() for w in _WORD.findall(passage)]
    for n in range(min(len(q), len(p)), 0, -1):
        grams = {tuple(p[i:i + n]) for i in range(len(p) - n + 1)}
        if any(tuple(q[i:i + n]) in grams for i in range(len(q) - n + 1)):
            return n
    return 0


# ------------------------------------------------------------ question set

@dataclass
class EvalQuestion:
    id: str
    question: str
    answer: str | None  # reference answer; None when the docs hold none
    answerable: bool
    evidence: list[Evidence] = field(default_factory=list)
    kind: str = "single"  # "single" | "multi" | "unanswerable"
    origin: str = "generated"  # "generated" | "edited" | "handwritten"
    candidate_id: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [e.as_dict() for e in self.evidence]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuestion":
        return cls(**{**d, "evidence": [Evidence(**e) for e in d.get("evidence", [])]})


def kind_of(evidence: list[Evidence]) -> str:
    """Derived from the evidence, never set by hand: an answer that needs two
    sections is multi-section whether or not anyone labelled it that way."""
    if not evidence:
        return "unanswerable"
    return "multi" if len({(e.doc_id, e.anchor) for e in evidence}) >= 2 else "single"


def load_questions(path: Path) -> list[EvalQuestion]:
    return [EvalQuestion.from_dict(json.loads(line))
            for line in path.open(encoding="utf-8") if line.strip()]


def save_questions(questions: list[EvalQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.as_dict(), ensure_ascii=False) + "\n")


def validate(questions: list[EvalQuestion], docs: dict[str, Document]) -> list[str]:
    """Every problem with the question set, as readable strings. Empty is good."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    for q in questions:
        where = f"{q.id}:"
        if q.id in seen_ids:
            problems.append(f"{where} duplicate id")
        seen_ids.add(q.id)
        norm = " ".join(q.question.lower().split())
        if not norm:
            problems.append(f"{where} empty question")
        elif norm in seen_text:
            problems.append(f"{where} same question as {seen_text[norm]}")
        seen_text.setdefault(norm, q.id)

        if q.answerable and not q.evidence:
            problems.append(f"{where} answerable but has no evidence")
        if q.answerable and not (q.answer or "").strip():
            problems.append(f"{where} answerable but has no reference answer")
        if not q.answerable and q.evidence:
            problems.append(f"{where} marked unanswerable but has evidence")
        if q.kind != kind_of(q.evidence):
            problems.append(f"{where} kind is {q.kind!r} but its evidence says {kind_of(q.evidence)!r}")

        for e in q.evidence:
            doc = docs.get(e.doc_id)
            if doc is None:
                problems.append(f"{where} evidence on unknown page {e.doc_id}")
                continue
            if not (0 <= e.start < e.end <= len(doc.text)):
                problems.append(f"{where} span {e.start}:{e.end} outside {e.doc_id}")
                continue
            if doc.text[e.start:e.end] != e.quote:
                problems.append(f"{where} quote no longer matches {e.doc_id}[{e.start}:{e.end}] "
                                "-- the corpus changed since the span was recorded")
    return problems
