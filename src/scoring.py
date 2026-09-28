"""Scoring retrieval against the question set's evidence spans.

A retrieved chunk is a **hit** when it covers at least half of one of the
question's evidence spans. Half, not any overlap: a chunk that brushes a
passage's first few characters has not handed the answer to the model. Every
evidence span in the set is shorter than every chunking's typical chunk
(median 101 characters, longest 413), so the rule is reachable under both
chunkings and favours neither.

    recall@5       a hit anywhere in the top 5 -- the plan's headline metric
    MRR@10         1 / rank of the first hit, 0 if none in the top 10
    all-evidence@5 every evidence span covered by the top 5; a question with
                   four supporting passages is not fully answerable from one
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .chunking import Chunk
from .evalset import Evidence

MIN_COVER = 0.5
K_RECALL = 5
K_MRR = 10

# Context budget for the equal-budget comparison. Chunks differ in size by
# chunking -- fixed ones are ~500 tokens, heading ones a median ~180 -- so "top
# 5" hands one scheme nearly three times the text of the other. Comparing hits
# within the same number of tokens asks the question generation will actually
# face: what fits in a fixed-size context. 2,500 is about five fixed chunks.
# Added after the first run showed the chunk-size gap; see NOTES.md.
BUDGET_TOKENS = 2_500


def covers(chunk: Chunk, ev: Evidence, min_fraction: float = MIN_COVER) -> bool:
    if chunk.doc_id != ev.doc_id:
        return False
    overlap = min(chunk.end, ev.end) - max(chunk.start, ev.start)
    if min_fraction <= 0:
        return overlap > 0
    return overlap >= min_fraction * (ev.end - ev.start)


@dataclass
class QuestionResult:
    qid: str
    kind: str
    top: list[str]  # chunk ids, best first
    first_hit: int | None  # 1-based rank of the first hit within the top K_MRR
    all_evidence_top5: bool
    latency_ms: float
    hit_any_overlap: bool = False  # recall@5 under the loosest rule, for comparison
    hit_budget: bool = False  # a hit within the first BUDGET_TOKENS of retrieved text
    budget_chunks: int = 0  # how many chunks that budget held

    @property
    def hit5(self) -> bool:
        return self.first_hit is not None and self.first_hit <= K_RECALL

    @property
    def rr(self) -> float:
        return 1.0 / self.first_hit if self.first_hit else 0.0


def within_budget(ranked: Sequence[Chunk], tokens: dict[str, int],
                  budget: int = BUDGET_TOKENS) -> list[Chunk]:
    """The leading chunks whose combined length fits in `budget` tokens
    (always at least the first one)."""
    out, used = [], 0
    for c in ranked:
        n = tokens[c.chunk_id]
        if out and used + n > budget:
            break
        out.append(c)
        used += n
    return out


def score_question(qid: str, kind: str, ranked: Sequence[Chunk], evidence: Sequence[Evidence],
                   latency_ms: float, tokens: dict[str, int] | None = None) -> QuestionResult:
    top = list(ranked[:K_MRR])
    first = next((r for r, c in enumerate(top, 1) if any(covers(c, e) for e in evidence)), None)
    top5 = top[:K_RECALL]
    budget = within_budget(ranked, tokens) if tokens else top5
    return QuestionResult(
        qid=qid, kind=kind, top=[c.chunk_id for c in top], first_hit=first,
        all_evidence_top5=all(any(covers(c, e) for c in top5) for e in evidence),
        latency_ms=latency_ms,
        hit_any_overlap=any(covers(c, e, 0.0) for c in top5 for e in evidence),
        hit_budget=any(covers(c, e) for c in budget for e in evidence),
        budget_chunks=len(budget),
    )


# ------------------------------------------------------------ statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; sane near 0 and 1, unlike the normal one."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def bootstrap_mean(values: Sequence[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: of the questions two runs disagree on, how
    lopsided is the split? Questions both get right or both get wrong carry no
    information about which is better, and are ignored."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], n: int = 10_000,
                           seed: int = 0) -> tuple[float, float, float]:
    """Mean of b - a over the same questions, with a 95% paired bootstrap interval."""
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    m = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(diffs) / m, boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


@dataclass
class Summary:
    n: int
    recall5: float
    recall5_ci: tuple[float, float]
    mrr10: float
    mrr10_ci: tuple[float, float]
    all_evidence5: float
    recall5_any_overlap: float
    recall_budget: float
    budget_chunks_mean: float
    latency_p50_ms: float
    latency_p95_ms: float
    extra: dict = field(default_factory=dict)


def summarize(results: Sequence[QuestionResult]) -> Summary:
    n = len(results)
    hits = sum(r.hit5 for r in results)
    lat = sorted(r.latency_ms for r in results)
    return Summary(
        n=n,
        recall5=hits / n, recall5_ci=wilson(hits, n),
        mrr10=sum(r.rr for r in results) / n, mrr10_ci=bootstrap_mean([r.rr for r in results]),
        all_evidence5=sum(r.all_evidence_top5 for r in results) / n,
        recall5_any_overlap=sum(r.hit_any_overlap for r in results) / n,
        recall_budget=sum(r.hit_budget for r in results) / n,
        budget_chunks_mean=sum(r.budget_chunks for r in results) / n,
        latency_p50_ms=lat[n // 2], latency_p95_ms=lat[min(n - 1, int(0.95 * n))],
    )
