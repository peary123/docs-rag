"""Review state for the question set: decisions, hand-written questions, search.

Kept free of any web code so the rules are testable: `scripts/03_review.py`
serves a page over these functions, and `scripts/04_build_questions.py` turns
the saved state into `eval/questions.jsonl`.

Every decision is saved with a timestamp to `eval/review_state.json`, so the
file is also the record of the review itself -- what was kept, what was
edited, what was dropped and why.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .corpus import Document
from .evalset import (EvalQuestion, Evidence, Section, kind_of, locate,
                      make_evidence, sections)

DROP_REASONS = {
    "too_easy": "Too easy: reuses the passage's wording",
    "ambiguous": "Ambiguous, or more than one right answer",
    "unsupported": "Answer not supported by the passage",
    "unrealistic": "Not something a user would ask",
    "duplicate": "Duplicate of another question",
    "other": "Other",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def candidates_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"candidates_digest": None, "decisions": {}, "handwritten": []}


def save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash mid-save never leaves half a review


# --------------------------------------------------------------- evidence

def evidence_from_selection(doc: Document, start: int, end: int, selected: str,
                            bounds: tuple[int, int] | None = None) -> Evidence:
    """Turn a span the reviewer selected in the browser into checked evidence.

    Offsets from a browser are the fragile part: JavaScript counts UTF-16 units
    and Python counts code points, and this corpus has emoji in it (the
    "Code above omitted 👆" markers). So the offsets are trusted only if the text
    at them is what the reviewer actually selected; otherwise the selected text
    is searched for within the section, and failing that it is an error.
    """
    lo, hi = bounds or (0, len(doc.text))
    want = selected.strip()
    if not want:
        raise ValueError("empty selection")
    if lo <= start < end <= hi and doc.text[start:end].strip() == want:
        # Trim the selection to its non-blank extent.
        start += len(doc.text[start:end]) - len(doc.text[start:end].lstrip())
        end = start + len(want)
        return make_evidence(doc, start, end)
    found = locate(doc.text, want, lo, hi)
    if found is None:
        raise ValueError("the selected text was not found in the section")
    return make_evidence(doc, *found)


def merge_evidence(evidence: list[Evidence], docs: dict[str, Document]) -> list[Evidence]:
    """Fold overlapping spans on the same page into one.

    A reviewer adding a selection that overlaps an existing span means "this
    region", not "these two regions"; keeping both would double-count it.

    Only true overlaps are merged, not spans that merely touch: two spans in
    adjacent sections can touch at the heading between them, and merging those
    would quietly turn a two-section question into a one-section one.
    """
    out: list[Evidence] = []
    for e in sorted(evidence, key=lambda e: (e.doc_id, e.start, e.end)):
        last = out[-1] if out else None
        if last and last.doc_id == e.doc_id and e.start < last.end:
            out[-1] = make_evidence(docs[e.doc_id], last.start, max(last.end, e.end))
        else:
            out.append(e)
    return out


def _same_evidence(a: list[dict], b: list[Evidence]) -> bool:
    return sorted((e["doc_id"], e["start"], e["end"]) for e in a) == \
        sorted((e.doc_id, e.start, e.end) for e in b)


# --------------------------------------------------------------- decisions

def decide(state: dict, candidate: dict, decision: str, *, question: str = "",
           answer: str = "", evidence: list[Evidence] | None = None,
           reason: str | None = None) -> dict:
    """Record keep or drop for one candidate. Deciding again replaces it."""
    cid = candidate["candidate_id"]
    if decision == "drop":
        if reason not in DROP_REASONS:
            raise ValueError(f"a drop needs one of: {', '.join(DROP_REASONS)}")
        entry = {"decision": "drop", "reason": reason, "at": _now()}
    elif decision == "keep":
        question, answer = question.strip(), answer.strip()
        evidence = evidence or []
        if not question or not answer:
            raise ValueError("a kept question needs both a question and an answer")
        if not evidence:
            raise ValueError("a kept question needs at least one evidence span")
        edited = (question != candidate["question"] or answer != candidate["answer"]
                  or not _same_evidence(candidate["evidence"], evidence))
        entry = {"decision": "keep", "question": question, "answer": answer,
                 "evidence": [e.as_dict() for e in evidence], "edited": edited, "at": _now()}
    else:
        raise ValueError("decision must be 'keep' or 'drop'")
    state["decisions"][cid] = entry
    return entry


def undo(state: dict, candidate_id: str) -> None:
    state["decisions"].pop(candidate_id, None)


def add_handwritten(state: dict, question: str, answer: str | None,
                    evidence: list[Evidence], note: str = "") -> dict:
    """A question the reviewer wrote. No evidence means the docs hold no answer."""
    question = question.strip()
    answer = (answer or "").strip() or None
    if not question:
        raise ValueError("empty question")
    if evidence and not answer:
        raise ValueError("a question with evidence needs a reference answer")
    if not evidence and answer:
        raise ValueError("an unanswerable question has no reference answer; "
                         "select evidence if the docs do answer it")
    used = {int(h["hid"][1:]) for h in state["handwritten"]}
    entry = {"hid": f"h{max(used, default=0) + 1:03d}", "question": question,
             "answer": answer, "evidence": [e.as_dict() for e in evidence],
             "note": note.strip(), "at": _now()}
    state["handwritten"].append(entry)
    return entry


def delete_handwritten(state: dict, hid: str) -> None:
    state["handwritten"] = [h for h in state["handwritten"] if h["hid"] != hid]


# --------------------------------------------------------------- building

def build_questions(candidates: list[dict], state: dict) -> list[EvalQuestion]:
    """The final question set: kept candidates first, then hand-written ones."""
    out: list[EvalQuestion] = []
    for c in candidates:
        d = state["decisions"].get(c["candidate_id"])
        if not d or d["decision"] != "keep":
            continue
        ev = [Evidence(**e) for e in d["evidence"]]
        out.append(EvalQuestion(
            id="", question=d["question"], answer=d["answer"], answerable=True,
            evidence=ev, kind=kind_of(ev),
            origin="edited" if d["edited"] else "generated",
            candidate_id=c["candidate_id"]))
    for h in state["handwritten"]:
        ev = [Evidence(**e) for e in h["evidence"]]
        out.append(EvalQuestion(
            id="", question=h["question"], answer=h["answer"], answerable=bool(ev),
            evidence=ev, kind=kind_of(ev), origin="handwritten"))
    for i, q in enumerate(out, 1):
        q.id = f"q{i:03d}"
    return out


# --------------------------------------------------------------- search

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = set("a an and are as at be by can do does for from how i in is it of on or "
            "the to what when where which why with you your".split())


class SectionSearch:
    """Keyword search over sections, for the reviewer -- not the RAG retriever.

    Its one job is to help check that a hand-written "unanswerable" question
    really has no answer in the docs, and to find evidence for a question the
    reviewer writes. Plain TF-IDF is enough for a person skimming ten results.
    """

    def __init__(self, docs: list[Document]) -> None:
        self.docs = {d.doc_id: d for d in docs}
        self.sections: list[Section] = [s for d in docs for s in sections(d)]
        self.tf = [Counter(self._tokens(s.text(self.docs[s.doc_id]))) for s in self.sections]
        df = Counter(t for tf in self.tf for t in tf)
        n = len(self.sections)
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]

    def search(self, query: str, k: int = 8) -> list[tuple[float, Section]]:
        terms = set(self._tokens(query))
        scored = []
        for s, tf in zip(self.sections, self.tf):
            score = sum(math.log(1 + tf[t]) * self.idf.get(t, 0.0) for t in terms if t in tf)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], x[1].doc_id, x[1].index))
        return scored[:k]
