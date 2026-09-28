"""Score the answers from scripts/06_generate_answers.py.

    correctness    gpt-4o judge against the reference answer, on the 116
                   answerable questions; a refusal is incorrect, unjudged
    faithfulness   gpt-4o judge, claim by claim against the excerpts the
                   answer was written from, on every rag answer that is not
                   a refusal
    refusals       string match on the refusal sentence -- no judge
    citations      does the answer cite an excerpt that covers the evidence --
                   span overlap, no judge

The one comparison decided in advance: correctness with retrieval against
closed-book, paired over the 116 answerable questions (McNemar). Correctness
split by whether the evidence reached the model is reported alongside, untested.

Judge prompts are the ones tested by scripts/07_validate_judge.py, unchanged.
Costs about $1 with gpt-4o, and is cached.

Usage:
    python scripts/08_score_answers.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config, corpus  # noqa: E402
from src.chunking import heading_chunks, model_tokenizer  # noqa: E402
from src.evalset import load_questions  # noqa: E402
from src.generation import format_context, strip_citations  # noqa: E402
from src.judge import JUDGE_MODEL, Judge  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.scoring import covers, mcnemar_exact, wilson  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
ANSWERS = config.RESULTS_DIR / "answers"
OUT_DIR = config.RESULTS_DIR / "judged"
SUMMARY = config.RESULTS_DIR / "generation_summary.json"
WORKERS = 8
DISCLAIMER = re.compile(r"\s*the (documentation|docs|excerpts?) (does|do) not (cover|provide|mention|include|specify)",
                        re.I)


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:5.1%}  [{lo:5.1%}, {hi:5.1%}]"


def main() -> int:
    docs = {d.doc_id: d for d in corpus.load_corpus()}
    questions = {q.id: q for q in load_questions(QUESTIONS)}
    chunks = {c.chunk_id: c for c in heading_chunks(list(docs.values()), model_tokenizer())}
    runs = {name: [json.loads(line) for line in (ANSWERS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
            for name in ("rag", "closed_book")}
    judge = Judge(LLMClient(model=JUDGE_MODEL))

    def correctness(row: dict) -> dict:
        q = questions[row["qid"]]
        if not q.answerable:
            return {}
        if row["refused"]:
            return {"verdict": "incorrect", "reason": "refused", "judged": False}
        v = judge.correctness(q.question, q.answer, strip_citations(row["text"]))
        return {"verdict": v.verdict, "reason": v.reason, "judged": True}

    def faithfulness(row: dict) -> dict:
        if row["refused"]:
            return {}
        context = format_context([chunks[i] for i in row["context"]], docs)
        f = judge.faithfulness(context, strip_citations(row["text"]))
        return {"faithful": f.faithful, "valid": f.valid, "claims": f.claims, "unsupported": f.unsupported}

    with ThreadPoolExecutor(WORKERS) as pool:
        judged = {name: list(pool.map(correctness, rows)) for name, rows in runs.items()}
        faith = list(pool.map(faithfulness, runs["rag"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in runs.items():
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                rec = {"qid": row["qid"], "answerable": row["answerable"], "refused": row["refused"],
                       "correctness": judged[name][i]}
                if name == "rag":
                    rec.update(evidence_in_context=row["evidence_in_context"], faithfulness=faith[i])
                rec["text"] = row["text"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    answerable = [q.id for q in questions.values() if q.answerable]
    unanswerable = [q.id for q in questions.values() if not q.answerable]
    idx = {name: {row["qid"]: i for i, row in enumerate(rows)} for name, rows in runs.items()}
    verdict = {name: {qid: judged[name][idx[name][qid]]["verdict"] for qid in answerable} for name in runs}
    summary: dict = {}

    print(f"correctness on {len(answerable)} answerable questions (judge: {JUDGE_MODEL})")
    for name in runs:
        counts = Counter(verdict[name].values())
        k = counts["correct"]
        print(f"  {name:12s} correct {rate(k, len(answerable))}   {dict(counts)}")
        summary[name] = {"correct": k, "n": len(answerable), "verdicts": dict(counts)}
    invalid = sum(v == "invalid" for name in runs for v in verdict[name].values())
    if invalid:
        print(f"  {invalid} judge replies could not be read")

    a = [verdict["closed_book"][q] == "correct" for q in answerable]
    b = [verdict["rag"][q] == "correct" for q in answerable]
    up, down = sum(not x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b))
    p = mcnemar_exact(down, up)
    print(f"  closed-book -> rag: {(sum(b) - sum(a)) / len(answerable):+.1%} "
          f"(+{up}/-{down}, McNemar p={p:.4f})  <- decided in advance")
    summary["rag_vs_closed_book"] = {"fixed": up, "broken": down, "mcnemar_p": p}

    rag_rows = {row["qid"]: row for row in runs["rag"]}
    print("\nrag correctness by whether the evidence reached the model (not tested):")
    summary["by_evidence"] = {}
    for label, want in (("evidence in context", True), ("evidence not in context", False)):
        ids = [q for q in answerable if rag_rows[q]["evidence_in_context"] == want]
        counts = Counter(verdict["rag"][q] for q in ids)
        refused = sum(rag_rows[q]["refused"] for q in ids)
        print(f"  {label:24s} correct {rate(counts['correct'], len(ids))}   {dict(counts)}; refused {refused}")
        summary["by_evidence"][label] = {"n": len(ids), "verdicts": dict(counts), "refused": refused}

    print("\nfaithfulness of rag answers that are not refusals:")
    fr = [(runs["rag"][i], f) for i, f in enumerate(faith) if f]
    ok = sum(f["faithful"] for _, f in fr)
    claims = sum(len(f["claims"]) for _, f in fr)
    unsupported = sum(len(f["unsupported"]) for _, f in fr)
    print(f"  every claim supported: {rate(ok, len(fr))}")
    print(f"  claims: {claims}, unsupported {unsupported} ({unsupported / max(1, claims):.1%})")
    for label, want in (("evidence in context", True), ("evidence not in context", False)):
        sub = [f for row, f in fr if row["answerable"] and row["evidence_in_context"] == want]
        print(f"  {label:24s} {rate(sum(f['faithful'] for f in sub), len(sub))}")
    if any(not f["valid"] for _, f in fr):
        print(f"  {sum(not f['valid'] for _, f in fr)} judge replies could not be read")
    # Added after the first run, on reading the unsupported claims: most were
    # the answer's own closing line about what "the documentation" does not
    # cover. Counted by pattern, so the split is reproducible, not a judgement.
    disclaimer_only = sum(1 for _, f in fr if not f["faithful"] and f["unsupported"]
                          and all(DISCLAIMER.match(u) for u in f["unsupported"]))
    print(f"  (post hoc) unfaithful only because of a line saying what the documentation "
          f"or excerpts do not cover: {disclaimer_only}")
    summary["faithfulness"] = {"answers": len(fr), "faithful": ok, "claims": claims, "unsupported": unsupported,
                               "unfaithful_only_by_disclaimer": disclaimer_only}

    print("\nrefusals (string match):")
    summary["refusals"] = {}
    for name in runs:
        rows = {row["qid"]: row for row in runs[name]}
        r_un = sum(rows[q]["refused"] for q in unanswerable)
        r_an = sum(rows[q]["refused"] for q in answerable)
        print(f"  {name:12s} refused {r_un}/{len(unanswerable)} unanswerable; "
              f"refused {r_an}/{len(answerable)} answerable")
        summary["refusals"][name] = {"unanswerable_refused": r_un, "n_unanswerable": len(unanswerable),
                                     "answerable_refused": r_an, "n_answerable": len(answerable)}

    # Citations: of the answered questions whose evidence was in the context,
    # does the answer cite an excerpt that actually holds it?
    eligible = [q for q in answerable if rag_rows[q]["evidence_in_context"] and not rag_rows[q]["refused"]]
    hits = sum(any(covers(chunks[c], e) for c in rag_rows[q]["cited_chunks"] for e in questions[q].evidence)
               for q in eligible)
    print(f"\ncitations: answer cites an excerpt holding the evidence in {rate(hits, len(eligible))}")
    summary["citations"] = {"n": len(eligible), "cites_evidence": hits}

    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    u = judge.llm.usage
    print(f"\njudge: {u.calls} calls sent, {u.cache_hits} from cache; "
          f"{u.input_tokens:,} input + {u.output_tokens:,} output tokens sent")
    print(f"verdicts in {OUT_DIR.relative_to(ROOT)}, summary in {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
