"""Tests for chunking, fusion and scoring -- the parts that decide the numbers.

No models are loaded: chunking takes the tokenizer as a function, so these use
a whitespace tokenizer, and the retrievers are exercised through `rrf` and the
scoring rules directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import Chunk, fixed_chunks, heading_chunks  # noqa: E402
from src.evalset import Evidence  # noqa: E402
from src.retrieval import bm25_tokens, rrf  # noqa: E402
from src.scoring import covers, mcnemar_exact, score_question, wilson  # noqa: E402
from tests.test_evalset import _doc  # noqa: E402


def words(text: str) -> list[tuple[int, int]]:
    """A stand-in tokenizer: one token per whitespace-separated word."""
    return [m.span() for m in re.finditer(r"\S+", text)]


LONG_PAGE = """# Guide { #guide }

## Empty parent { #parent }

## Child { #child }

""" + "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(6)) + "\n\n## Short { #short }\n\nJust a line.\n"


# ------------------------------------------------------------------ chunking

def test_fixed_windows_overlap_and_stay_within_a_page() -> None:
    doc = _doc(LONG_PAGE, "guide.md")
    chunks = fixed_chunks([doc], words, size=40, overlap=10)
    assert len(chunks) > 3
    for a, b in zip(chunks, chunks[1:]):
        assert a.start < b.start < a.end  # consecutive windows overlap
    assert all(len(words(c.text)) <= 40 for c in chunks)
    assert chunks[-1].end == len(doc.text.rstrip())


def test_heading_chunks_fold_empty_sections_and_split_long_ones() -> None:
    doc = _doc(LONG_PAGE, "guide.md")
    chunks = heading_chunks([doc], words, max_tokens=80)
    # the heading-only "Empty parent" section is not a chunk of its own
    assert not any(c.text.strip() == "## Empty parent" for c in chunks)
    assert any("## Empty parent" in c.text and "Paragraph 0" in c.text for c in chunks)
    # the long "Child" section is split, every piece within budget
    assert all(len(words(c.text)) <= 80 for c in chunks)
    assert sum("Paragraph" in c.text for c in chunks) >= 3
    assert chunks[-1].text.startswith("## Short")


def test_chunk_text_is_exactly_its_span() -> None:
    doc = _doc(LONG_PAGE, "guide.md")
    for c in fixed_chunks([doc], words, 40, 10) + heading_chunks([doc], words, 80):
        assert c.text == doc.text[c.start:c.end]


# ------------------------------------------------------------------ retrieval pieces

def test_bm25_keeps_identifiers_whole() -> None:
    assert bm25_tokens("Set response_model=None on HTTPException") == \
        ["set", "response_model", "none", "on", "httpexception"]


def test_rrf_uses_ranks_and_breaks_ties_deterministically() -> None:
    # 7 is first in one list and absent from the other; 3 is second in both.
    fused = rrf([[7, 3, 5], [4, 3, 9]], k=60)
    assert fused[0] == 3  # consistently near the top beats a single first place
    assert fused == rrf([[7, 3, 5], [4, 3, 9]], k=60)
    assert set(fused) == {7, 3, 5, 4, 9}


# ------------------------------------------------------------------ scoring

def _chunk(start: int, end: int, doc: str = "a.md") -> Chunk:
    return Chunk(f"c:{start}", doc, start, end, "", "x" * (end - start))


def test_a_chunk_must_cover_half_the_evidence() -> None:
    ev = Evidence("a.md", 100, 200, "", "x" * 100)
    assert covers(_chunk(0, 150), ev)          # covers 50 of 100
    assert not covers(_chunk(0, 149), ev)      # covers 49
    assert covers(_chunk(0, 101), ev, 0.0)     # any overlap, under the loose rule
    assert not covers(_chunk(100, 200, "b.md"), ev)  # right offsets, wrong page


def test_rank_is_the_first_hit_and_all_evidence_needs_every_span() -> None:
    ev = [Evidence("a.md", 100, 120, "", ""), Evidence("a.md", 500, 520, "", "")]
    ranked = [_chunk(0, 50), _chunk(90, 130), _chunk(600, 700), _chunk(480, 530)]
    r = score_question("q1", "single", ranked, ev, 1.0)
    assert r.first_hit == 2 and r.hit5 and r.rr == 0.5
    assert r.all_evidence_top5
    only_first = score_question("q1", "single", ranked[:3], ev, 1.0)
    assert only_first.hit5 and not only_first.all_evidence_top5


def test_statistics() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) < 0.01
    lo, hi = wilson(80, 100)
    assert lo < 0.8 < hi and 0.70 < lo and hi < 0.88
