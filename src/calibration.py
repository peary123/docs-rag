"""Answers whose verdict is known in advance, for testing the judges.

An LLM judge's scores mean something only if the judge agrees with the truth.
The usual check is to grade a sample of answers by hand and measure agreement.
This project checks the judges against answers built so that the right verdict
is known by construction, from each question's reference answer:

    for the correctness judge
      paraphrase     the reference, reworded                  should be: correct
      substitution   the reference with one fact changed      should be: not correct
      swap           the reference answer of the most         should be: not correct
                     similar *other* question
    for the faithfulness judge (against the retrieved excerpts)
      supported      the reference answer                     should be: faithful
      fabricated     the reference plus one invented,         should be: unfaithful
                     specific sentence the excerpts lack

Paraphrases test whether a judge is too strict -- penalising wording. The other
four test whether it is too lenient, which is the failure that inflates a
headline number.

This is weaker than hand-grading: a planted error is one kind of error, and
real ones can be subtler. And the labels are only as good as the construction
-- a paraphrase can drop a fact, a "wrong" substitution can turn out harmless --
so each construction is checked mechanically where it can be, each probe records
exactly what was changed, and every disagreement is written out to read. What
it offers instead is scale: hundreds of probes, not thirty answers.

The probes are written by gpt-4o-mini, the model that answers: they then read
like the answers being judged, and the judge (gpt-4o) is never grading its own
writing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLMClient

PROBE_MODEL = "gpt-4o-mini"

EXPECTED = {
    "paraphrase": "correct",
    "substitution": "not correct",
    "swap": "not correct",
    "supported": "faithful",
    "fabricated": "unfaithful",
}
CORRECTNESS_PROBES = ("paraphrase", "substitution", "swap")
FAITHFULNESS_PROBES = ("supported", "fabricated")


@dataclass
class Probe:
    qid: str
    kind: str  # a key of EXPECTED
    answer: str
    detail: dict = field(default_factory=dict)  # what was changed, for reading disagreements

    @property
    def probe_id(self) -> str:
        return f"{self.qid}:{self.kind}"

    @property
    def expected(self) -> str:
        return EXPECTED[self.kind]


# ----------------------------------------------------------------- paraphrase

PARAPHRASE_PROMPT = """\
Rewrite the answer below so that it says exactly the same thing in different words: change the sentence structure and the wording of the prose. Keep every fact. Keep every name, identifier, number and piece of code exactly as written. Do not add anything, and do not leave anything out.

Answer:
{answer}

Reply with a JSON object: {{"answer": "<the rewritten answer>"}}"""

# Anything a paraphrase must carry over unchanged: code, identifiers, numbers.
_CODE_TOKEN = re.compile(
    r"`[^`]+`"                       # inline code
    r"|\b\w+(?:\.\w+)+\b"            # dotted names: app.include_router
    r"|\b\w+\("                      # calls: Depends(
    r"|\b\w+="                       # keyword arguments: response_model=
    r"|\b[a-z0-9]+_\w+\b"            # snake_case
    r"|--[\w-]+"                     # command-line flags
    r"|@\w+"                         # decorators
    r"|\b[A-Z][a-z0-9]+[A-Z]\w*\b"   # CamelCase: BaseSettings, FastAPI
    r"|\b\d+(?:\.\d+)*\b"            # numbers and versions
)
_NOT_CODE = {"e.g", "i.e"}


def code_tokens(text: str) -> set[str]:
    return {t for t in _CODE_TOKEN.findall(text) if t.lower() not in _NOT_CODE}


def _json_field(text: str, key: str) -> str | None:
    try:
        value = json.loads(text)[key]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def check_paraphrase(reference: str, rewritten: str | None) -> str:
    """Why a paraphrase can't be used as a known-correct answer ('' if it can)."""
    if not rewritten:
        return "no answer returned"
    if " ".join(rewritten.split()) == " ".join(reference.split()):
        return "unchanged"
    missing = code_tokens(reference) - code_tokens(rewritten)
    if missing:
        return "lost " + ", ".join(sorted(missing))
    return ""


def paraphrase(llm: LLMClient, qid: str, reference: str) -> tuple[Probe | None, str]:
    resp = llm.complete(PARAPHRASE_PROMPT.format(answer=reference), json_mode=True, max_tokens=600)
    rewritten = _json_field(resp.text, "answer")
    problem = check_paraphrase(reference, rewritten)
    if problem:
        return None, problem
    return Probe(qid, "paraphrase", rewritten), ""


# --------------------------------------------------------------- substitution

SUBSTITUTION_PROMPT = """\
Below is a question about FastAPI and a correct answer to it. Make the answer wrong by changing one fact that a user would rely on: a function, parameter or class name, a value or default, a number, whether something is possible or required, or which of two options to use. After the change the answer must be wrong about FastAPI, but still sound plausible to someone who does not know FastAPI well.

Question:
{question}

Answer:
{answer}

Reply with a JSON object: {{"original": "<a short phrase copied exactly from the answer>", "replacement": "<the phrase to put in its place>", "why_wrong": "<one sentence>"}}"""

MAX_ORIGINAL_CHARS = 80


def apply_substitution(reference: str, original: str | None, replacement: str | None) -> tuple[str | None, str]:
    """The reference with one phrase replaced, or why that can't be done.

    The phrase must occur exactly once, so the change is exactly the one the
    model described and nothing else moves.
    """
    if not original or not replacement:
        return None, "incomplete reply"
    if len(original) > MAX_ORIGINAL_CHARS:
        return None, "phrase too long to be one fact"
    if original.strip().lower() == replacement.strip().lower():
        return None, "replacement is the same phrase"
    count = reference.count(original)
    if count != 1:
        return None, f"phrase occurs {count} times"
    return reference.replace(original, replacement, 1), ""


def substitute(llm: LLMClient, qid: str, question: str, reference: str) -> tuple[Probe | None, str]:
    resp = llm.complete(SUBSTITUTION_PROMPT.format(question=question, answer=reference),
                        json_mode=True, max_tokens=300)
    original = _json_field(resp.text, "original")
    replacement = _json_field(resp.text, "replacement")
    changed, problem = apply_substitution(reference, original, replacement)
    if changed is None:
        return None, problem
    return Probe(qid, "substitution", changed, {
        "original": original, "replacement": replacement,
        "why_wrong": _json_field(resp.text, "why_wrong") or ""}), ""


# ----------------------------------------------------------------------- swap

def swap(qid: str, reference: str, other_qid: str, other_reference: str,
         similarity: float) -> tuple[Probe | None, str]:
    if " ".join(other_reference.split()) == " ".join(reference.split()):
        return None, "same reference answer"
    return Probe(qid, "swap", other_reference, {"from": other_qid, "similarity": round(similarity, 3)}), ""


# ---------------------------------------------------------------- faithfulness

FABRICATION_PROMPT = """\
Below are documentation excerpts and an answer that is supported by them. Write one sentence to add to the end of the answer. It must sound as if it belongs there, and it must state something specific -- a name, a value, a default, a limit or a behaviour -- that the excerpts do not say. It does not matter whether the sentence is true about FastAPI; what matters is that the excerpts do not support it.

Excerpts:

{context}

Answer:
{answer}

Reply with a JSON object: {{"sentence": "<the sentence to add>"}}"""


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def check_fabrication(context: str, sentence: str | None) -> str:
    if not sentence:
        return "no sentence returned"
    if "\n" in sentence:
        return "more than one line"
    if _squash(sentence).rstrip(".") in _squash(context):
        return "sentence is in the excerpts"
    return ""


def fabricate(llm: LLMClient, qid: str, context: str, reference: str) -> tuple[Probe | None, str]:
    resp = llm.complete(FABRICATION_PROMPT.format(context=context, answer=reference),
                        json_mode=True, max_tokens=200)
    sentence = _json_field(resp.text, "sentence")
    problem = check_fabrication(context, sentence)
    if problem:
        return None, problem
    return Probe(qid, "fabricated", f"{reference.rstrip()} {sentence}", {"added": sentence}), ""
