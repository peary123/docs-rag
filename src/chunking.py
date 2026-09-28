"""Two ways to cut the rendered pages into retrievable chunks.

* **fixed**: windows of 500 tokens overlapping by 50, within one page.
* **headings**: one chunk per section, with a section too long for the
  embedding model split at paragraph boundaries.

Tokens are counted with the embedding model's own tokenizer. bge-small reads at
most 512 tokens (two of them special), so a "500-token" chunk counted any other
way could silently lose its tail inside the dense retriever -- and a chunk the
dense retriever only half reads is a result about truncation, not retrieval.

Every chunk keeps its character span in the page, because that is how it is
scored: a chunk is a hit when it covers a question's evidence span, whichever
scheme produced it.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass

from . import config
from .corpus import Document
from .evalset import sections

MAX_TOKENS = 500
OVERLAP = 50

# text -> (start, end) character offsets of each token
Tokenize = Callable[[str], list[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    anchor: str  # section the chunk starts in -- what an answer built on it cites
    text: str


@functools.lru_cache(maxsize=1)
def model_tokenizer() -> Tokenize:
    """The embedding model's tokenizer, as a text -> token offsets function."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL)

    def offsets(text: str) -> list[tuple[int, int]]:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        return [tuple(o) for o in enc["offset_mapping"]]

    return offsets


def _make(scheme: str, doc: Document, n: int, start: int, end: int) -> Chunk:
    heading = doc.section_at(start)
    return Chunk(f"{scheme}:{doc.doc_id}:{n}", doc.doc_id, start, end,
                 heading.anchor if heading else "", doc.text[start:end])


def _windows(spans: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Character ranges of token windows: `size` tokens, stepping size - overlap."""
    if not spans:
        return []
    step = size - overlap
    out = []
    for i in range(0, len(spans), step):
        window = spans[i:i + size]
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(spans):
            break
    return out


def fixed_chunks(docs: list[Document], tokenize: Tokenize,
                 size: int = MAX_TOKENS, overlap: int = OVERLAP) -> list[Chunk]:
    """Token windows over each page. A window never crosses into another page."""
    out: list[Chunk] = []
    for doc in docs:
        for n, (start, end) in enumerate(_windows(tokenize(doc.text), size, overlap)):
            out.append(_make("fixed", doc, n, start, end))
    return out


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _paragraphs(text: str, base: int) -> list[tuple[int, int]]:
    """Character ranges of blank-line-separated paragraphs, offset by `base`."""
    out, pos = [], 0
    for m in _PARAGRAPH_BREAK.finditer(text):
        if m.start() > pos:
            out.append((base + pos, base + m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((base + pos, base + len(text)))
    return out


def heading_chunks(docs: list[Document], tokenize: Tokenize,
                   max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """One chunk per section; long sections are split at paragraph breaks.

    A section that is nothing but its heading -- a parent heading followed
    directly by a child heading -- is folded into the next section instead of
    becoming a chunk of its own. A near-empty chunk made entirely of title words
    is exactly what keyword search over-ranks, and it holds no answer.
    """
    out: list[Chunk] = []
    for doc in docs:
        n = 0
        pending_start: int | None = None  # a heading-only section waiting to be folded in
        for sec in sections(doc):
            body = doc.text[sec.start:sec.end].split("\n", 1)
            if len(body) < 2 or not body[1].strip():
                pending_start = sec.start if pending_start is None else pending_start
                continue
            start = sec.start if pending_start is None else pending_start
            pending_start = None
            text = doc.text[start:sec.end]
            if len(tokenize(text)) <= max_tokens:
                out.append(_make("headings", doc, n, start, sec.end))
                n += 1
                continue
            # Too long for the embedding model: pack paragraphs greedily, and
            # cut a single oversized paragraph (usually a long code block) into
            # token windows.
            cur_start = cur_end = None
            cur_tokens = 0
            for p_start, p_end in _paragraphs(text, start):
                p_tokens = len(tokenize(doc.text[p_start:p_end]))
                if cur_start is not None and cur_tokens + p_tokens > max_tokens:
                    out.append(_make("headings", doc, n, cur_start, cur_end))
                    n += 1
                    cur_start = None
                if p_tokens > max_tokens:
                    spans = [(p_start + a, p_start + b) for a, b in tokenize(doc.text[p_start:p_end])]
                    for a, b in _windows(spans, max_tokens, OVERLAP):
                        out.append(_make("headings", doc, n, a, b))
                        n += 1
                    continue
                if cur_start is None:
                    cur_start, cur_tokens = p_start, 0
                cur_end = p_end
                cur_tokens += p_tokens
            if cur_start is not None:
                out.append(_make("headings", doc, n, cur_start, cur_end))
                n += 1
        if pending_start is not None:  # the page ends on a heading with no body
            out.append(_make("headings", doc, n, pending_start, len(doc.text)))
    return out


CHUNKERS: dict[str, Callable[[list[Document], Tokenize], list[Chunk]]] = {
    "fixed": fixed_chunks,
    "headings": heading_chunks,
}
