"""Tests for building the question set and recording its review.

What these guard against is a question set that is quietly wrong: evidence that
points at the wrong words, a two-section question that has become a
one-section one, a reviewer's edit that is recorded as untouched. Each of those
would still produce numbers -- just not meaningful ones.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, corpus  # noqa: E402
from src import review as rv  # noqa: E402
from src.evalset import (  # noqa: E402
    EXCLUDED_SECTIONS, EvalQuestion, Evidence, kind_of, locate, longest_shared_run,
    make_evidence, sample_sections, sections, validate,
)
from src.evalset import term_hits as rv_term_hits  # noqa: E402

HAVE_DATA = config.DOCS_DIR.exists()

PAGE = """# Query parameters { #query-parameters }

Declare function parameters that are not part of the path and they become query parameters.

## Defaults { #defaults }

The same way, you can declare optional query parameters, by setting their default to `None`:

```python
async def read_item(q: str | None = None):
    return {"q": q}
```

## Required { #required }

```python
# Code above omitted 👆
q: str
```

Click the button "Try it out" and **FastAPI** will **extract** the value for you.
"""


def _doc(markdown: str = PAGE, doc_id: str = "tutorial/query-params.md") -> corpus.Document:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs" / "en").mkdir(parents=True)
        return corpus.render(doc_id, markdown, build_dir=root / "docs" / "en", root=root)


def _span(doc: corpus.Document, text: str) -> Evidence:
    i = doc.text.index(text)
    return make_evidence(doc, i, i + len(text))


# ------------------------------------------------------------------ sections

def test_sections_run_from_each_heading_to_the_next() -> None:
    doc = _doc()
    ss = sections(doc)
    assert [s.anchor for s in ss] == ["query-parameters", "defaults", "required"]
    assert all(a.end == b.start for a, b in zip(ss, ss[1:]))
    assert ss[-1].end == len(doc.text)
    assert "default to `None`" in ss[1].text(doc)


# ------------------------------------------------------------------ locate

def test_locate_finds_exact_and_reflowed_text() -> None:
    doc = _doc()
    exact = "by setting their default to `None`"
    i = doc.text.index(exact)
    assert locate(doc.text, exact) == (i, i + len(exact))
    assert locate(doc.text, "by   setting\ntheir default") == (i, i + len("by setting their default"))


def test_locate_ignores_quote_style_emphasis_and_backticks() -> None:
    """A model writing JSON swaps " for ' and drops ** -- the words survive."""
    doc = _doc()
    assert locate(doc.text, "Click the button 'Try it out' and FastAPI will extract")
    assert locate(doc.text, "by setting their default to None")
    assert locate(doc.text, 'return {"q": q}') and locate(doc.text, "return {'q': q}")


def test_locate_accepts_elision_but_not_changed_words() -> None:
    doc = _doc()
    found = locate(doc.text, "The same way, ... default to `None`")
    assert found and doc.text[found[0]:found[1]].startswith("The same way")
    assert locate(doc.text, "by setting their value to None") is None
    assert locate(doc.text, "The same way ... a sentence that is not there") is None


def test_locate_respects_section_bounds() -> None:
    doc = _doc()
    required = sections(doc)[2]
    assert locate(doc.text, "default to `None`", required.start, required.end) is None


# ------------------------------------------------------------------ overlap, kind

def test_longest_shared_run_counts_consecutive_words() -> None:
    passage = "you can declare optional query parameters by setting their default"
    assert longest_shared_run("How do I declare optional query parameters?", passage) == 4
    assert longest_shared_run("What makes an argument not required?", passage) == 0


def test_kind_follows_the_evidence() -> None:
    doc = _doc()
    a, b = _span(doc, "default to `None`"), _span(doc, "Try it out")
    assert kind_of([]) == "unanswerable"
    assert kind_of([a]) == "single"
    assert kind_of([a, _span(doc, "optional query parameters")]) == "single"
    assert kind_of([a, b]) == "multi"


# ------------------------------------------------------------------ validate

def _question(doc: corpus.Document, **kw) -> EvalQuestion:
    ev = [_span(doc, "default to `None`")]
    base = dict(id="q001", question="How do I make a query parameter optional?",
                answer="Give it a default of None.", answerable=True, evidence=ev, kind="single")
    return EvalQuestion(**{**base, **kw})


def test_a_sound_question_set_validates() -> None:
    doc = _doc()
    unanswerable = EvalQuestion(id="q002", question="Does FastAPI ship an ORM?", answer=None,
                                answerable=False, kind="unanswerable", origin="handwritten")
    assert validate([_question(doc), unanswerable], {doc.doc_id: doc}) == []


def test_validate_catches_drift_and_broken_rules() -> None:
    doc = _doc()
    docs = {doc.doc_id: doc}
    drifted = _question(doc)
    drifted.evidence[0].quote = "something the corpus no longer says"
    assert any("no longer matches" in p for p in validate([drifted], docs))
    assert validate([_question(doc, evidence=[], kind="unanswerable")], docs)  # answerable, no evidence
    assert validate([_question(doc, answerable=False, answer=None)], docs)     # unanswerable, with evidence
    assert validate([_question(doc, kind="multi")], docs)                      # kind contradicts evidence
    dup = [_question(doc), _question(doc, id="q002", question="how do i make a query parameter  OPTIONAL?")]
    assert any("same question" in p for p in validate(dup, docs))


# ------------------------------------------------------------------ review

def _candidate(doc: corpus.Document) -> dict:
    return {"candidate_id": "c001", "question": "How do I make a query parameter optional?",
            "answer": "Give it a default of None.",
            "evidence": [_span(doc, "default to `None`").as_dict()]}


def _state() -> dict:
    return {"candidates_digest": None, "decisions": {}, "handwritten": []}


def test_keeping_unchanged_is_not_recorded_as_an_edit() -> None:
    doc, state = _doc(), _state()
    cand = _candidate(doc)
    same = [Evidence(**cand["evidence"][0])]
    assert rv.decide(state, cand, "keep", question=cand["question"],
                     answer=cand["answer"], evidence=same)["edited"] is False
    assert rv.decide(state, cand, "keep", question="Can a query parameter be left out?",
                     answer=cand["answer"], evidence=same)["edited"] is True


def test_a_drop_needs_a_reason_and_a_keep_needs_evidence() -> None:
    doc, state = _doc(), _state()
    cand = _candidate(doc)
    for bad in (lambda: rv.decide(state, cand, "drop"),
                lambda: rv.decide(state, cand, "keep", question="q", answer="a", evidence=[])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")
    rv.decide(state, cand, "drop", reason="too_easy")
    assert state["decisions"]["c001"]["reason"] == "too_easy"


def test_handwritten_questions_are_answerable_exactly_when_they_have_evidence() -> None:
    doc, state = _doc(), _state()
    rv.add_handwritten(state, "Does FastAPI include an ORM?", None, [])
    rv.add_handwritten(state, "Optional query params?", "Default of None.", [_span(doc, "default to `None`")])
    for bad in (lambda: rv.add_handwritten(state, "q", "an answer", []),
                lambda: rv.add_handwritten(state, "q", None, [_span(doc, "Try it out")])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")
    built = rv.build_questions([], state)
    assert [(q.id, q.answerable, q.kind, q.origin) for q in built] == [
        ("q001", False, "unanswerable", "handwritten"), ("q002", True, "single", "handwritten")]


def test_build_keeps_only_kept_candidates_in_candidate_order() -> None:
    doc, state = _doc(), _state()
    c1, c2 = _candidate(doc), {**_candidate(doc), "candidate_id": "c002"}
    rv.decide(state, c2, "keep", question=c2["question"], answer=c2["answer"],
              evidence=[Evidence(**c2["evidence"][0])])
    rv.decide(state, c1, "drop", reason="duplicate")
    built = rv.build_questions([c1, c2], state)
    assert [(q.id, q.candidate_id, q.origin) for q in built] == [("q001", "c002", "generated")]


def test_overlapping_selections_merge_but_touching_sections_stay_apart() -> None:
    doc = _doc()
    a = _span(doc, "by setting their default")
    b = _span(doc, "their default to `None`")
    merged = rv.merge_evidence([a, b], {doc.doc_id: doc})
    assert len(merged) == 1 and merged[0].quote == "by setting their default to `None`"
    ss = sections(doc)
    left = make_evidence(doc, ss[1].end - 5, ss[1].end)   # ends where "Required" begins
    right = make_evidence(doc, ss[2].start, ss[2].start + 5)
    kept = rv.merge_evidence([left, right], {doc.doc_id: doc})
    assert len(kept) == 2 and kind_of(kept) == "multi"


def test_a_browser_selection_after_an_emoji_still_lands_on_the_right_words() -> None:
    """JavaScript counts an emoji as two units, Python as one: offsets from the
    page can be one off for every emoji before the selection."""
    doc = _doc()
    target = "Try it out"
    i = doc.text.index(target)
    ev = rv.evidence_from_selection(doc, i + 1, i + 1 + len(target), target)
    assert (ev.start, ev.end, ev.quote) == (i, i + len(target), target)
    try:
        rv.evidence_from_selection(doc, i, i + 5, "words that are not on the page")
    except ValueError:
        return
    raise AssertionError("a selection that is not on the page must be refused")


def test_term_hits_match_at_word_starts_only() -> None:
    """A substring search for "aws" finds "flaws"; that must not count."""
    doc = _doc(PAGE + "\nSome tutorials have security flaws and throttling.\n")
    docs = {doc.doc_id: doc}
    hits = rv_term_hits(docs, ["aws", "throttl", "flaws", "kafka"])
    assert hits == {"aws": [], "throttl": [doc.doc_id], "flaws": [doc.doc_id], "kafka": []}


def test_unanswerable_questions_come_last_and_never_renumber_the_rest() -> None:
    doc, state = _doc(), _state()
    cand = _candidate(doc)
    rv.decide(state, cand, "keep", question=cand["question"], answer=cand["answer"],
              evidence=[Evidence(**cand["evidence"][0])])
    before = rv.build_questions([cand], state)
    after = rv.build_questions([cand], state, [{"question": "Does FastAPI include an ORM?"}])
    assert [q.id for q in before] == ["q001"]
    assert after[0].as_dict() == before[0].as_dict()
    assert (after[1].id, after[1].answerable, after[1].kind, after[1].origin) == \
        ("q002", False, "unanswerable", "written")
    assert validate(after, {doc.doc_id: doc}) == []


def test_section_search_ranks_the_matching_section_first() -> None:
    doc = _doc()
    top = rv.SectionSearch([doc]).search("optional default None")
    assert top and top[0][1].anchor == "defaults"


# ------------------------------------------------------------------ real corpus

def test_sampling_spreads_across_pages_and_skips_excluded_sections() -> None:
    if not HAVE_DATA:
        return
    docs = corpus.load_corpus()
    picked = sample_sections(docs, 120, seed=1)
    per_page: dict[str, int] = {}
    for s in picked:
        per_page[s.doc_id] = per_page.get(s.doc_id, 0) + 1
    assert max(per_page.values()) <= 2
    assert not any(s.key in EXCLUDED_SECTIONS for s in picked)
    assert [s.key for s in picked] == [s.key for s in sample_sections(docs, 120, seed=1)]
