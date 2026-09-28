"""Retrieval baseline: two chunkings x four retrievers, scored on the question set.

Runs every answerable question through each combination and reports recall@5
and MRR@10 against the evidence spans, with 95% intervals. The comparisons are
fixed in advance (below), not picked after looking at the table:

    within a chunking:  hybrid vs bm25, hybrid vs dense, hybrid+rerank vs hybrid
    across chunkings:   headings vs fixed, for each retriever

Every setting was chosen before any of this ran and none is tuned on the
question set: 500-token chunks with 50 overlap, RRF k = 60, rerank depth 20.

Runs locally on the CPU; costs nothing. First run downloads two small models
and encodes both chunkings (a few minutes); later runs reuse the cached vectors.

Usage:
    python scripts/05_retrieval_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config, corpus  # noqa: E402
from src.chunking import CHUNKERS, model_tokenizer  # noqa: E402
from src.evalset import load_questions  # noqa: E402
from src.retrieval import RERANK_DEPTH, build  # noqa: E402
from src.scoring import (  # noqa: E402
    BUDGET_TOKENS, K_MRR, mcnemar_exact, paired_bootstrap_delta, score_question, summarize,
)

QUESTIONS = ROOT / "eval" / "questions.jsonl"
OUT_DIR = config.RESULTS_DIR / "retrieval"
METHODS = ["bm25", "dense", "hybrid", "hybrid+rerank"]
WITHIN = [("bm25", "hybrid"), ("dense", "hybrid"), ("hybrid", "hybrid+rerank")]


def main() -> int:
    docs = corpus.load_corpus()
    questions = [q for q in load_questions(QUESTIONS) if q.answerable]
    tokenize = model_tokenizer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(questions)} answerable questions, {len(docs)} pages\n")

    results: dict[tuple[str, str], list] = {}
    stats: dict[tuple[str, str], object] = {}
    for scheme, chunker in CHUNKERS.items():
        chunks = chunker(docs, tokenize)
        tokens = {c.chunk_id: len(tokenize(c.text)) for c in chunks}
        started = time.perf_counter()
        retrievers = build(chunks)
        print(f"{scheme}: {len(chunks)} chunks, index built in {time.perf_counter() - started:.1f}s")
        for method in METHODS:
            r = retrievers[method]
            r.search("warm up the models before timing anything", K_MRR)
            rows = []
            for q in questions:
                t0 = time.perf_counter()
                # 20 deep, the most the reranker returns, so the equal-budget
                # column can reach past the top 10 for the smaller chunks.
                ranked = [chunks[i] for i in r.search(q.question, RERANK_DEPTH)]
                ms = (time.perf_counter() - t0) * 1000
                rows.append(score_question(q.id, q.kind, ranked, q.evidence, ms, tokens))
            results[(scheme, method)] = rows
            stats[(scheme, method)] = summarize(rows)
            with (OUT_DIR / f"{scheme}__{method.replace('+', '_')}.jsonl").open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps({**row.__dict__, "hit5": row.hit5}, ensure_ascii=False) + "\n")

    print(f"\n{'chunking':9s} {'retriever':14s} {'recall@5':>9s} {'95% CI':>15s} {'MRR@10':>7s} "
          f"{'95% CI':>15s} {'all-ev@5':>9s} {'any-ovl@5':>10s} {'@2.5k tok':>10s} {'chunks':>6s} {'p50 ms':>7s}")
    for (scheme, method), s in stats.items():
        print(f"{scheme:9s} {method:14s} {s.recall5:9.1%} "
              f"[{s.recall5_ci[0]:5.1%}, {s.recall5_ci[1]:5.1%}] {s.mrr10:7.3f} "
              f"[{s.mrr10_ci[0]:.3f}, {s.mrr10_ci[1]:.3f}] {s.all_evidence5:9.1%} "
              f"{s.recall5_any_overlap:10.1%} {s.recall_budget:10.1%} {s.budget_chunks_mean:6.1f} {s.latency_p50_ms:7.1f}")

    print("\npaired comparisons (fixed in advance):")
    comparisons = [(scheme, a, scheme, b) for scheme in CHUNKERS for a, b in WITHIN]
    comparisons += [("fixed", m, "headings", m) for m in METHODS]
    report = []
    for s1, m1, s2, m2 in comparisons:
        a, b = results[(s1, m1)], results[(s2, m2)]
        fixed_ = sum(not x.hit5 and y.hit5 for x, y in zip(a, b))
        broken = sum(x.hit5 and not y.hit5 for x, y in zip(a, b))
        p = mcnemar_exact(broken, fixed_)
        d, lo, hi = paired_bootstrap_delta([x.rr for x in a], [y.rr for y in b])
        label = f"{s1}/{m1} -> {s2}/{m2}"
        print(f"  {label:42s} recall@5 {(sum(y.hit5 for y in b) - sum(x.hit5 for x in a)) / len(a):+6.1%} "
              f"(+{fixed_}/-{broken}, McNemar p={p:.3f})   MRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        report.append({"from": f"{s1}/{m1}", "to": f"{s2}/{m2}", "fixed": fixed_, "broken": broken,
                       "mcnemar_p": round(p, 4), "mrr_delta": round(d, 4), "mrr_ci": [round(lo, 4), round(hi, 4)]})

    print(f"\nequal context budget ({BUDGET_TOKENS} tokens) -- added after the first run, to check whether")
    print("fixed chunks win only because five of them hold ~3x the text of five heading chunks:")
    for m in METHODS:
        a, b = results[("fixed", m)], results[("headings", m)]
        up = sum(not x.hit_budget and y.hit_budget for x, y in zip(a, b))
        down = sum(x.hit_budget and not y.hit_budget for x, y in zip(a, b))
        diff = (sum(y.hit_budget for y in b) - sum(x.hit_budget for x in a)) / len(a)
        print(f"  fixed/{m} -> headings/{m}".ljust(46) + f"{diff:+6.1%} (+{up}/-{down}, McNemar p={mcnemar_exact(down, up):.3f})")
        report.append({"from": f"fixed/{m}", "to": f"headings/{m}", "metric": f"recall@{BUDGET_TOKENS}tok",
                       "fixed": up, "broken": down, "mcnemar_p": round(mcnemar_exact(down, up), 4)})

    summary = {f"{s}/{m}": {k: (list(v) if isinstance(v, tuple) else v) for k, v in st.__dict__.items()}
               for (s, m), st in stats.items()}
    (config.RESULTS_DIR / "retrieval_summary.json").write_text(
        json.dumps({"summary": summary, "comparisons": report}, indent=2), encoding="utf-8")
    print(f"\nper-question records in {OUT_DIR.relative_to(ROOT)}, summary in results/retrieval_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
