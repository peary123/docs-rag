"""Reading answers and judge replies, and building known-verdict probes.

None of this calls a model: every function here is the deterministic part
around a model call, which is where a silent scoring bug would live.
"""

import json

from src.calibration import (
    apply_substitution, check_fabrication, check_paraphrase, code_tokens, swap,
)
from src.chunking import Chunk
from src.corpus import Document, Heading
from src.generation import (
    CLOSED_BOOK_REFUSAL, REFUSAL, citations, format_context, is_refusal,
    mentions_refusal, strip_citations,
)
from src.judge import parse_correctness, parse_faithfulness


# ------------------------------------------------------------------ refusals

def test_refusal_is_the_exact_sentence():
    assert is_refusal(REFUSAL)
    assert is_refusal("  The documentation does not cover this.\n")


def test_refusal_tolerates_wrapping_but_not_words():
    assert is_refusal('"The documentation does not cover this."')
    assert is_refusal("**The documentation does not cover this**")
    assert is_refusal("the documentation does not cover this")
    assert not is_refusal("The documentation does not cover this, but you could try a middleware.")
    assert not is_refusal("The docs don't cover this.")


def test_partial_refusal_is_counted_separately():
    text = "The documentation does not cover this. However, you can use Depends [1]."
    assert not is_refusal(text)
    assert mentions_refusal(text)
    assert not mentions_refusal(REFUSAL)


def test_closed_book_uses_its_own_sentence():
    assert is_refusal("I don't know.", CLOSED_BOOK_REFUSAL)
    assert not is_refusal("I don't know.")


# ----------------------------------------------------------------- citations

def test_citations_in_order_of_first_use():
    assert citations("Use Depends [2]. It runs first [1][2]. Also [3, 1].") == [2, 1, 3]


def test_indexing_and_code_are_not_citations():
    text = "Read items[0] from the list [1].\n```python\nx = data[2]\n```\nThen `y[3]` [4]."
    assert citations(text) == [1, 4]


def test_strip_citations_leaves_clean_prose_and_code():
    text = "Use Depends [2]. It runs first [1][3], always.\n```python\nx = [1]\n```"
    assert strip_citations(text) == "Use Depends. It runs first, always.\n```python\nx = [1]\n```"


# ------------------------------------------------------------------ context

def _doc():
    text = "# Title\n\nIntro.\n\n## Order matters { #order }\n\nPut fixed paths first.\n"
    return Document(
        doc_id="tutorial/path-params.md", url="https://fastapi.tiangolo.com/tutorial/path-params/",
        title="Path Parameters", text=text,
        headings=(Heading(1, "Title", "title", 0), Heading(2, "Order matters", "order", 17)),
        includes=0, included_chars=0)


def test_context_numbers_excerpts_and_links_sections():
    doc = _doc()
    chunks = [Chunk("h:0", doc.doc_id, 17, len(doc.text), "order", doc.text[17:]),
              Chunk("h:1", doc.doc_id, 0, 17, "title", doc.text[:17])]
    ctx = format_context(chunks, {doc.doc_id: doc})
    assert ctx.startswith("[1] Path Parameters > Order matters\n"
                          "https://fastapi.tiangolo.com/tutorial/path-params/#order\n")
    assert "\n\n[2] Path Parameters\nhttps://fastapi.tiangolo.com/tutorial/path-params/\n# Title" in ctx


# -------------------------------------------------------------------- judges

def test_correctness_verdict_parsed_and_validated():
    ok = parse_correctness(json.dumps({"reason": "matches", "verdict": "Correct"}))
    assert ok.verdict == "correct" and ok.correct
    assert parse_correctness(json.dumps({"verdict": "mostly right"})).verdict == "invalid"
    assert parse_correctness("not json").verdict == "invalid"


def test_faithful_needs_every_claim_supported():
    good = parse_faithfulness(json.dumps({"claims": [{"claim": "a", "supported": True}]}))
    bad = parse_faithfulness(json.dumps({"claims": [{"claim": "a", "supported": True},
                                                    {"claim": "b", "supported": False}]}))
    assert good.faithful and not bad.faithful and bad.unsupported == ["b"]


def test_no_claims_or_unreadable_is_not_faithful():
    assert not parse_faithfulness(json.dumps({"claims": []})).faithful
    assert not parse_faithfulness("{}").faithful
    fuzzy = parse_faithfulness(json.dumps({"claims": [{"claim": "a", "supported": "yes"}]}))
    assert not fuzzy.valid and not fuzzy.faithful


# ------------------------------------------------------------------- probes

def test_code_tokens_catch_identifiers_not_abbreviations():
    toks = code_tokens("Use app.include_router(graphql_app, prefix=...), e.g. with BaseSettings and --reload on port 8000.")
    assert {"app.include_router", "graphql_app", "prefix=", "BaseSettings", "--reload", "8000"} <= toks
    assert "e.g" not in toks


def test_paraphrase_rejected_when_it_drops_an_identifier():
    ref = "Set response_model=None to disable it."
    assert check_paraphrase(ref, "To turn it off, pass response_model=None.") == ""
    assert check_paraphrase(ref, "To turn it off, pass None as the response model.").startswith("lost")
    assert check_paraphrase(ref, ref) == "unchanged"
    assert check_paraphrase(ref, None) == "no answer returned"


def test_substitution_changes_exactly_one_phrase():
    ref = "Use Header, otherwise the parameter is read as a query parameter."
    changed, problem = apply_substitution(ref, "Header", "Cookie")
    assert problem == "" and changed == "Use Cookie, otherwise the parameter is read as a query parameter."
    assert apply_substitution(ref, "parameter", "field")[1] == "phrase occurs 2 times"
    assert apply_substitution(ref, "Body", "Form")[1] == "phrase occurs 0 times"
    assert apply_substitution(ref, "Header", "header")[1] == "replacement is the same phrase"
    assert apply_substitution(ref, None, "x")[1] == "incomplete reply"


def test_fabrication_must_be_new_and_one_line():
    ctx = "[1] Page\nThe default port is 8000."
    assert check_fabrication(ctx, "The default timeout is 30 seconds.") == ""
    assert check_fabrication(ctx, "The default port is 8000.") == "sentence is in the excerpts"
    assert check_fabrication(ctx, "One.\nTwo.") == "more than one line"


def test_swap_refuses_an_identical_answer():
    assert swap("q1", "Same answer.", "q2", "Same  answer.", 0.9)[0] is None
    probe, _ = swap("q1", "One answer.", "q2", "Another answer.", 0.9)
    assert probe.answer == "Another answer." and probe.expected == "not correct"
