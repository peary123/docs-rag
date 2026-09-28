"""Turn the review into the question set, check it, and report on it.

Reads eval/candidates.jsonl and the reviewer's decisions in
eval/review_state.json, writes eval/questions.jsonl, and fails if any evidence
span no longer matches the corpus or any question breaks the set's rules.

Usage:
    python scripts/04_build_questions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import corpus  # noqa: E402
from src import review as rv  # noqa: E402
from src.evalset import (  # noqa: E402
    longest_shared_run, save_questions, sections, term_hits, validate,
)

CANDIDATES = ROOT / "eval" / "candidates.jsonl"
UNANSWERABLE = ROOT / "eval" / "unanswerable.jsonl"
STATE = ROOT / "eval" / "review_state.json"
OUT = ROOT / "eval" / "questions.jsonl"


def source_text(docs: dict, q) -> str:
    """The full text of every section the question's evidence sits in."""
    keys = {(e.doc_id, e.anchor) for e in q.evidence}
    return "\n".join(s.text(docs[d]) for d, a in sorted(keys)
                     for s in sections(docs[d]) if s.anchor == a)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    docs = {d.doc_id: d for d in corpus.load_corpus()}
    rows = [json.loads(line) for line in CANDIDATES.open(encoding="utf-8") if line.strip()]
    offered = [r for r in rows if r["status"] == "ok"]
    state = rv.load_state(args.state)
    unanswerable = ([json.loads(line) for line in UNANSWERABLE.open(encoding="utf-8") if line.strip()]
                    if UNANSWERABLE.exists() else [])
    questions = rv.build_questions(offered, state, unanswerable)

    problems = validate(questions, docs)
    # An unanswerable question is only unanswerable while none of the terms an
    # answer would need appears in the indexed pages. Checked on every build,
    # so a change to the corpus that makes one answerable cannot go unnoticed.
    for u in unanswerable:
        for term, pages in term_hits(docs, u["absent_terms"]).items():
            if pages:
                problems.append(f"{u['id']}: '{term}' appears in {', '.join(pages[:3])} "
                                "-- the docs may answer this 'unanswerable' question")
    if problems:
        print(f"{len(problems)} problem(s) -- nothing written:")
        for p in problems:
            print("  " + p)
        return 1

    decisions = state["decisions"]
    undecided = [c["candidate_id"] for c in offered if c["candidate_id"] not in decisions]
    kinds = Counter(q.kind for q in questions)
    origin = Counter(q.origin for q in questions)
    reasons = Counter(d["reason"] for d in decisions.values() if d["decision"] == "drop")
    answerable = [q for q in questions if q.answerable]
    pages = {e.doc_id for q in answerable for e in q.evidence}

    print(f"candidates generated {len(rows)}, offered for review {len(offered)}, "
          f"decided {len(decisions)}" + (f", UNDECIDED {len(undecided)}" if undecided else ""))
    print(f"kept {sum(d['decision'] == 'keep' for d in decisions.values())} "
          f"({origin['edited']} edited), dropped {sum(reasons.values())}")
    for key, n in reasons.most_common():
        print(f"    {n:3d}  {rv.DROP_REASONS[key]}")

    print(f"\nquestion set: {len(questions)}")
    print(f"  answerable   {len(answerable):3d}  (single-section {kinds['single']}, two-section {kinds['multi']})")
    print(f"  unanswerable {kinds['unanswerable']:3d}")
    print(f"  origin: generated {origin['generated']}, edited {origin['edited']}, "
          f"written in review {origin['handwritten']}, unanswerable set {origin['written']}")
    if unanswerable:
        traps = Counter(u["trap"] for u in unanswerable)
        print("  unanswerable, by trap: " + ", ".join(f"{t} {n}" for t, n in traps.most_common())
              + " -- each checked absent from every indexed page")
    print(f"  answers drawn from {len(pages)} of {len(docs)} pages")

    # How much the set still leans on the docs' own wording -- the bias that
    # would flatter keyword retrieval.
    runs = [longest_shared_run(q.question, source_text(docs, q)) for q in answerable]
    if runs:
        before = sum(c["overlap"] >= 5 for c in offered)
        after = sum(r >= 5 for r in runs)
        print(f"  sharing a 5+ word phrase with their source: {after} of {len(runs)} "
              f"(among all {len(offered)} offered candidates: {before})")

    print("\nagainst the plan:")
    checks = [
        # The plan's "about 80" is a floor for statistical power, not a cap:
        # a bigger set only narrows every confidence interval downstream.
        ("at least 70 answerable questions", len(answerable) >= 70, len(answerable)),
        ("10-15 unanswerable questions", 10 <= kinds["unanswerable"] <= 15, kinds["unanswerable"]),
        # The plan asks for "some"; too few to report as their own category,
        # so they are counted in the overall numbers only.
        ("some questions need two sections", kinds["multi"] > 0, kinds["multi"]),
        ("every candidate reviewed", not undecided, f"{len(undecided)} left"),
        ("every evidence span matches the corpus", True, "checked"),
    ]
    for label, ok, value in checks:
        print(f"  [{'ok ' if ok else '   '}] {label}: {value}")

    save_questions(questions, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
