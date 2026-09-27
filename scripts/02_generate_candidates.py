"""Generate candidate questions for the evaluation set, for a person to review.

Samples sections across the docs, asks a model to write one realistic question
per section (or per pair of nearby sections, for questions that need both),
and keeps a candidate only if every evidence quote it gives can be found in the
source text. Nothing here is final: `scripts/03_review.py` is where each
candidate is kept, edited or dropped by hand.

The generator is a stronger model (gpt-4o) than the one that will answer the
questions (gpt-4o-mini). Models tend to favour answers phrased the way they
would phrase them, so writing the questions with the answering model would tilt
the evaluation towards it.

Usage:
    python scripts/02_generate_candidates.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config, corpus  # noqa: E402
from src.evalset import (  # noqa: E402
    Evidence, Section, locate, longest_shared_run, make_evidence,
    sample_section_pairs, sample_sections,
)
from src.llm import LLMClient  # noqa: E402

GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o")
N_SINGLE = 120
N_PAIRS = 20
SEED = 20260926
OUT = ROOT / "eval" / "candidates.jsonl"

# USD per million tokens, for the cost line only.
PRICE = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}

SYSTEM = (
    "You write evaluation questions for a question-answering system built on "
    "the FastAPI documentation. You reply with a single JSON object."
)

RULES = """Rules:
- Ask it the way a developer who has NOT read this page would ask: in your own words.
  Do not reuse the section's headings or distinctive phrases.
- It must have one clear, correct answer, fully supported by the text below.
- Do not mention "the section", "the page", "the docs" or "the text".
- Prefer how-to and why questions over asking what something is called.
- If the text cannot support a realistic question (a list of links, a
  stub, a heading over nothing), return {"skip": true} instead."""

SINGLE_PROMPT = """Below is one section of the FastAPI documentation.
Page: {title}
Section: {heading}

Write ONE question a FastAPI user might realistically ask that this section answers.

{rules}

Return JSON:
{{"question": "...",
  "answer": "1-3 sentences, only what the section supports",
  "evidence": ["a quote copied character for character from the section", "..."]}}
Each evidence quote must be copied exactly from the section, at most two
sentences or three lines of code. Give one to three quotes.

Section:
<<<
{text}
>>>"""

PAIR_PROMPT = """Below are two sections from the same page of the FastAPI documentation.
Page: {title}

Write ONE question a FastAPI user might realistically ask whose full answer
needs information from BOTH sections -- not a question either section answers
alone.

{rules}

Return JSON:
{{"question": "...",
  "answer": "1-3 sentences, only what the two sections support",
  "evidence": [{{"section": 1, "quote": "..."}}, {{"section": 2, "quote": "..."}}]}}
Include at least one quote from each section, each copied character for
character, at most two sentences or three lines of code.

Section 1 -- {heading1}
<<<
{text1}
>>>

Section 2 -- {heading2}
<<<
{text2}
>>>"""


def _ask(client: LLMClient, prompt: str) -> tuple[dict | None, str]:
    response = client.complete(prompt, system=SYSTEM, json_mode=True, max_tokens=700)
    try:
        data = json.loads(response.text)
        return (data if isinstance(data, dict) else None), response.text
    except json.JSONDecodeError:
        return None, response.text


def _finish(cid: str, kind: str, secs: list[Section], data: dict | None, raw: str,
            spans: list[Evidence] | None, passage: str) -> dict:
    record = {
        "candidate_id": cid,
        "kind": kind,
        "sections": [s.key for s in secs],
        "model": GEN_MODEL,
    }
    if data is None:
        return {**record, "status": "bad_json", "raw": raw}
    if data.get("skip"):
        return {**record, "status": "skipped_by_model"}
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    if not question or not answer:
        return {**record, "status": "incomplete", "raw": raw}
    if spans is None:
        # At least one quote could not be found in the source text. The model
        # paraphrased where it was told to copy, so its evidence -- and possibly
        # its answer -- is not what the page says. Rejected, not repaired.
        return {**record, "status": "evidence_not_verbatim", "question": question, "raw": raw}
    return {
        **record,
        "status": "ok",
        "question": question,
        "answer": answer,
        "evidence": [e.as_dict() for e in spans],
        "overlap": longest_shared_run(question, passage),
        # The model's own output, so every evidence match can be audited
        # against what was actually quoted.
        "raw": raw,
    }


def single(client: LLMClient, cid: str, doc: corpus.Document, sec: Section) -> dict:
    text = sec.text(doc)
    data, raw = _ask(client, SINGLE_PROMPT.format(
        title=doc.title, heading=sec.title, rules=RULES, text=text))
    spans: list[Evidence] | None = []
    for quote in (data or {}).get("evidence", []) or []:
        found = locate(doc.text, str(quote), sec.start, sec.end)
        if found is None:
            spans = None
            break
        spans.append(make_evidence(doc, *found))
    if spans == []:
        spans = None  # an answer with no evidence at all is not checkable
    return _finish(cid, "single", [sec], data, raw, spans, text)


def pair(client: LLMClient, cid: str, doc: corpus.Document, a: Section, b: Section) -> dict:
    data, raw = _ask(client, PAIR_PROMPT.format(
        title=doc.title, rules=RULES, heading1=a.title, text1=a.text(doc),
        heading2=b.title, text2=b.text(doc)))
    spans: list[Evidence] | None = []
    touched: set[int] = set()
    for item in (data or {}).get("evidence", []) or []:
        quote = str(item.get("quote", "")) if isinstance(item, dict) else str(item)
        found = None
        for n, sec in ((1, a), (2, b)):
            found = locate(doc.text, quote, sec.start, sec.end)
            if found:
                touched.add(n)
                break
        if found is None:
            spans = None
            break
        spans.append(make_evidence(doc, *found))
    if spans is not None and touched != {1, 2}:
        spans = None  # it claims to need both sections but quotes only one
    return _finish(cid, "multi", [a, b], data, raw, spans, a.text(doc) + "\n" + b.text(doc))


def main() -> int:
    docs = {d.doc_id: d for d in corpus.load_corpus()}
    singles = sample_sections(list(docs.values()), N_SINGLE, SEED)
    pairs = sample_section_pairs(list(docs.values()), N_PAIRS, SEED + 1,
                                 avoid={s.key for s in singles})
    client = LLMClient(model=GEN_MODEL)

    jobs = [(f"c{i:03d}", "single", (s,)) for i, s in enumerate(singles, 1)]
    jobs += [(f"c{i:03d}", "multi", p) for i, p in enumerate(pairs, len(singles) + 1)]
    print(f"generating {len(jobs)} candidates with {GEN_MODEL} "
          f"({len(singles)} single-section, {len(pairs)} two-section)\n")

    def run(job):
        cid, kind, secs = job
        doc = docs[secs[0].doc_id]
        return single(client, cid, doc, secs[0]) if kind == "single" else pair(client, cid, doc, *secs)

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(run, jobs))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    status = Counter((r["kind"], r["status"]) for r in records)
    print("candidate funnel:")
    for kind in ("single", "multi"):
        total = sum(n for (k, _), n in status.items() if k == kind)
        parts = ", ".join(f"{s} {n}" for (k, s), n in sorted(status.items()) if k == kind)
        print(f"  {kind:6s} {total:3d}  ->  {parts}")

    ok = [r for r in records if r["status"] == "ok"]
    runs = Counter(min(r["overlap"], 8) for r in ok)
    print(f"\noffered for review: {len(ok)}")
    print("longest phrase shared with the source, in words (8 = 8 or more):")
    print("  " + "  ".join(f"{k}:{runs.get(k, 0)}" for k in range(9)))
    copied = sum(r["overlap"] >= 5 for r in ok)
    print(f"  {copied} candidates share 5+ consecutive words with their source")

    u = client.usage
    p_in, p_out = PRICE.get(GEN_MODEL, (0.0, 0.0))
    cost = (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6
    print(f"\n{u.calls} API calls, {u.cache_hits} cache hits, ~${cost:.2f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
