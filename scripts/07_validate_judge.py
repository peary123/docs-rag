"""Test the two judges on answers whose right verdict is known.

Builds five kinds of probe from the reference answers (see src/calibration.py),
checks each mechanically, and has the gpt-4o judges grade them:

    paraphrase     the reference, reworded              should be judged correct
    substitution   one fact changed                     should not be judged correct
    swap           the most similar other question's    should not be judged correct
                   reference answer
    supported      the reference, against the excerpts  should be judged faithful
    fabricated     the reference plus one invented      should not be judged faithful
                   sentence, against the excerpts

The faithfulness probes use the excerpts the system actually retrieved for the
question, and only for questions where every evidence span is among them -- so
the reference answer is backed by what the judge sees.

Run after scripts/06_generate_answers.py (it needs the retrieved excerpts).
Probes are written by gpt-4o-mini (cents); judging them with gpt-4o costs about
$2. Both are cached.

Usage:
    python scripts/07_validate_judge.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from src import calibration, config, corpus  # noqa: E402
from src.calibration import CORRECTNESS_PROBES, FAITHFULNESS_PROBES, PROBE_MODEL  # noqa: E402
from src.chunking import heading_chunks, model_tokenizer  # noqa: E402
from src.evalset import load_questions  # noqa: E402
from src.generation import format_context  # noqa: E402
from src.judge import JUDGE_MODEL, Judge  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieval import _sentence_model  # noqa: E402
from src.scoring import wilson  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RAG_ANSWERS = config.RESULTS_DIR / "answers" / "rag.jsonl"
PROBES = ROOT / "eval" / "judge_probes.jsonl"
OUT = config.RESULTS_DIR / "judge_validation.jsonl"
SUMMARY = config.RESULTS_DIR / "judge_validation_summary.json"
WORKERS = 8


def nearest_other(texts: list[str]) -> list[tuple[int, float]]:
    """For each text, the most similar other text (bge cosine) and its similarity."""
    vecs = _sentence_model(config.EMBED_MODEL).encode(
        texts, normalize_embeddings=True, show_progress_bar=False)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1.0)
    return [(int(j), float(sims[i, j])) for i, j in enumerate(sims.argmax(axis=1))]


def main() -> int:
    docs = {d.doc_id: d for d in corpus.load_corpus()}
    questions = [q for q in load_questions(QUESTIONS) if q.answerable]
    chunks = {c.chunk_id: c for c in heading_chunks(list(docs.values()), model_tokenizer())}
    rag = {r["qid"]: r for r in map(json.loads, RAG_ANSWERS.read_text(encoding="utf-8").splitlines())}
    contexts = {q.id: format_context([chunks[i] for i in rag[q.id]["context"]], docs) for q in questions}
    writer = LLMClient(model=PROBE_MODEL)

    # ---- build the probes
    rejected: dict[str, Counter] = {k: Counter() for k in calibration.EXPECTED}
    probes: list[calibration.Probe] = []

    def keep(kind: str, result: tuple) -> None:
        probe, problem = result
        if probe is None:
            rejected[kind]["lost code or identifiers" if problem.startswith("lost") else problem] += 1
        else:
            probes.append(probe)

    backed = [q for q in questions if rag[q.id]["all_evidence_in_context"]]
    with ThreadPoolExecutor(WORKERS) as pool:
        para = list(pool.map(lambda q: calibration.paraphrase(writer, q.id, q.answer), questions))
        subs = list(pool.map(lambda q: calibration.substitute(writer, q.id, q.question, q.answer), questions))
        fabs = list(pool.map(lambda q: calibration.fabricate(writer, q.id, contexts[q.id], q.answer), backed))
    neighbours = nearest_other([q.question for q in questions])
    for q, p, s, (j, sim) in zip(questions, para, subs, neighbours):
        keep("paraphrase", p)
        keep("substitution", s)
        keep("swap", calibration.swap(q.id, q.answer, questions[j].id, questions[j].answer, sim))
    for q, f in zip(backed, fabs):
        probes.append(calibration.Probe(q.id, "supported", q.answer))
        keep("fabricated", f)

    probes.sort(key=lambda p: (p.qid, list(calibration.EXPECTED).index(p.kind)))
    with PROBES.open("w", encoding="utf-8") as fh:
        for p in probes:
            fh.write(json.dumps({"probe_id": p.probe_id, "qid": p.qid, "kind": p.kind,
                                 "expected": p.expected, "answer": p.answer, "detail": p.detail},
                                ensure_ascii=False) + "\n")

    # ---- judge them
    judge = Judge(LLMClient(model=JUDGE_MODEL))
    by_id = {q.id: q for q in questions}

    def grade(p: calibration.Probe) -> dict:
        q = by_id[p.qid]
        if p.kind in CORRECTNESS_PROBES:
            v = judge.correctness(q.question, q.answer, p.answer)
            return {"verdict": v.verdict, "reason": v.reason,
                    "as_expected": v.correct if p.expected == "correct" else
                    (not v.correct and v.verdict != "invalid")}
        f = judge.faithfulness(contexts[p.qid], p.answer)
        return {"verdict": ("faithful" if f.faithful else "unfaithful") if f.valid else "invalid",
                "claims": f.claims, "unsupported": f.unsupported,
                "as_expected": f.valid and (f.faithful == (p.expected == "faithful"))}

    with ThreadPoolExecutor(WORKERS) as pool:
        graded = list(pool.map(grade, probes))
    with OUT.open("w", encoding="utf-8") as fh:
        for p, g in zip(probes, graded):
            fh.write(json.dumps({"probe_id": p.probe_id, "kind": p.kind, "expected": p.expected, **g,
                                 "answer": p.answer, "detail": p.detail}, ensure_ascii=False) + "\n")

    # ---- report
    print(f"{len(questions)} answerable questions; {len(backed)} have all their evidence in the "
          f"retrieved excerpts and get faithfulness probes\n")
    print(f"{'probe':13s} {'expected':12s} {'built':>5s} {'rejected':>8s} {'as expected':>12s} {'95% CI':>14s}  verdicts")
    summary = {}
    for kind in calibration.EXPECTED:
        rows = [g for p, g in zip(probes, graded) if p.kind == kind]
        ok = sum(g["as_expected"] for g in rows)
        lo, hi = wilson(ok, len(rows))
        verdicts = Counter(g["verdict"] for g in rows)
        print(f"{kind:13s} {calibration.EXPECTED[kind]:12s} {len(rows):5d} {sum(rejected[kind].values()):8d} "
              f"{ok:5d} {ok / len(rows):6.1%} [{lo:5.1%}, {hi:5.1%}]  {dict(verdicts)}")
        summary[kind] = {"expected": calibration.EXPECTED[kind], "n": len(rows), "as_expected": ok,
                         "rate": ok / len(rows), "ci": [lo, hi], "verdicts": dict(verdicts),
                         "rejected": dict(rejected[kind])}
    for kind, reasons in rejected.items():
        if reasons:
            print(f"  {kind} rejected by the construction checks: {dict(reasons)}")

    # Paired: the same answer with and without the invented sentence. Catching
    # the sentence means passing the clean answer and failing the planted one.
    pair = {}
    for p, g in zip(probes, graded):
        if p.kind in FAITHFULNESS_PROBES:
            pair.setdefault(p.qid, {})[p.kind] = g["verdict"]
    both = [v for v in pair.values() if len(v) == 2]
    clean = [v for v in both if v["supported"] == "faithful"]
    caught = sum(v["fabricated"] == "unfaithful" for v in clean)
    print(f"\nfaithfulness pairs: {len(both)}; clean answer passed in {len(clean)}; "
          f"of those, the invented sentence caught in {caught} ({caught / max(1, len(clean)):.1%})")
    summary["fabricated_paired"] = {"pairs": len(both), "clean_passed": len(clean), "caught": caught}

    too_lenient = ("substitution", "swap", "fabricated")
    wrong_total = sum(summary[k]["n"] for k in too_lenient)
    wrong_caught = sum(summary[k]["as_expected"] for k in too_lenient)
    print(f"planted errors rejected: {wrong_caught}/{wrong_total} ({wrong_caught / wrong_total:.1%}); "
          f"correct answers accepted: paraphrase {summary['paraphrase']['as_expected']}/{summary['paraphrase']['n']}, "
          f"supported {summary['supported']['as_expected']}/{summary['supported']['n']}")
    summary["planted_errors"] = {"n": wrong_total, "rejected": wrong_caught}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, client in (("probe writer", writer), ("judge", judge.llm)):
        u = client.usage
        print(f"{name}: {u.calls} calls sent, {u.cache_hits} from cache; "
              f"{u.input_tokens:,} input + {u.output_tokens:,} output tokens sent")
    print(f"probes in {PROBES.relative_to(ROOT)}, verdicts in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
