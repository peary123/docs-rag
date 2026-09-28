"""Answer every question twice: from retrieved docs, and closed-book.

    rag          heading chunks, hybrid retrieval, the excerpts that fit in
                 2,500 tokens, gpt-4o-mini told to answer only from them,
                 cite them, and say "The documentation does not cover this."
                 when they don't answer the question
    closed_book  the same model and question with no excerpts -- how much
                 it can answer from memory, since it was trained on an
                 older version of these docs

All 128 questions, answerable and not. Answers are written to
results/answers/; judging them is scripts/08_score_answers.py.

Retrieval is re-run here and checked against the retrieval comparison: the
share of questions whose evidence reaches the model must equal the
headings/hybrid recall at 2,500 tokens reported there, or the two stages are
not measuring the same system.

Costs a few cents with gpt-4o-mini, and is cached.

Usage:
    python scripts/06_generate_answers.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config, corpus  # noqa: E402
from src.chunking import heading_chunks, model_tokenizer  # noqa: E402
from src.evalset import load_questions  # noqa: E402
from src.generation import ANSWER_MODEL, Generator, closed_book  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieval import build  # noqa: E402
from src.scoring import covers  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
OUT_DIR = config.RESULTS_DIR / "answers"
RETRIEVAL_SUMMARY = config.RESULTS_DIR / "retrieval_summary.json"
WORKERS = 8


def main() -> int:
    docs = corpus.load_corpus()
    by_doc = {d.doc_id: d for d in docs}
    questions = load_questions(QUESTIONS)
    tokenize = model_tokenizer()
    chunks = heading_chunks(docs, tokenize)
    tokens = {c.chunk_id: len(tokenize(c.text)) for c in chunks}
    retriever = build(chunks)["hybrid"]
    llm = LLMClient(model=ANSWER_MODEL)
    gen = Generator(llm, retriever, chunks, tokens, by_doc)
    print(f"{len(questions)} questions, {len(chunks)} heading chunks, answering with {ANSWER_MODEL}\n")

    # Retrieval first, in one thread: the embedding model is not shared across threads.
    retriever.search("warm up the model before timing anything", 5)
    contexts, facts = {}, {}
    for q in questions:
        t0 = time.perf_counter()
        ctx = gen.retrieve(q.question)
        ms = (time.perf_counter() - t0) * 1000
        contexts[q.id] = ctx
        facts[q.id] = {
            "retrieval_ms": round(ms, 1),
            "context_tokens": sum(tokens[c.chunk_id] for c in ctx),
            "evidence_in_context": q.answerable and any(covers(c, e) for c in ctx for e in q.evidence),
            "all_evidence_in_context": q.answerable and all(any(covers(c, e) for c in ctx) for e in q.evidence),
        }

    answerable = [q for q in questions if q.answerable]
    in_ctx = sum(facts[q.id]["evidence_in_context"] for q in answerable) / len(answerable)
    reported = json.loads(RETRIEVAL_SUMMARY.read_text(encoding="utf-8"))["summary"]["headings/hybrid"]["recall_budget"]
    print(f"evidence reaches the model for {in_ctx:.1%} of answerable questions "
          f"(retrieval comparison reported {reported:.1%})")
    if abs(in_ctx - reported) > 1e-9:
        print("  MISMATCH: generation is not using the retrieval that was measured")
        return 1

    with ThreadPoolExecutor(WORKERS) as pool:
        rag = list(pool.map(lambda q: gen.answer(q.id, q.question, contexts[q.id]), questions))
        cb = list(pool.map(lambda q: closed_book(llm, q.id, q.question), questions))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, answers in (("rag", rag), ("closed_book", cb)):
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for q, a in zip(questions, answers):
                row = {"qid": q.id, "answerable": q.answerable, "kind": q.kind, "question": q.question,
                       **{k: v for k, v in a.__dict__.items() if k != "qid"}}
                if name == "rag":
                    row.update(facts[q.id], cited_chunks=a.cited_chunks)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    unanswerable = [q for q in questions if not q.answerable]
    for name, answers in (("rag", rag), ("closed_book", cb)):
        a_by = {a.qid: a for a in answers}
        print(f"\n{name}:")
        print(f"  refused {sum(a_by[q.id].refused for q in unanswerable)}/{len(unanswerable)} unanswerable, "
              f"{sum(a_by[q.id].refused for q in answerable)}/{len(answerable)} answerable")
        print(f"  partial refusals (refusal sentence inside a longer answer): "
              f"{sum(a.partial_refusal for a in answers)}")
        if name == "rag":
            answered = [a for a in answers if not a.refused]
            print(f"  answers citing at least one excerpt: {sum(bool(a.citations) for a in answered)}/{len(answered)}")
            print(f"  citations to excerpts that do not exist: {sum(len(a.invalid_citations) for a in answers)}")
            print(f"  mean context: {sum(f['context_tokens'] for f in facts.values()) / len(facts):.0f} tokens, "
                  f"{sum(len(c) for c in contexts.values()) / len(contexts):.1f} excerpts")
    u = llm.usage
    print(f"\n{u.calls} calls sent, {u.cache_hits} from cache; "
          f"{u.input_tokens:,} input + {u.output_tokens:,} output tokens sent")
    print(f"answers in {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
