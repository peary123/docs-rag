"""Two LLM judges: is an answer correct, and is it supported by its context.

They answer different questions and can disagree, which is the point:

* **correctness** compares the answer with the reference answer. It cannot tell
  an answer copied from the excerpts from one the model remembered.
* **faithfulness** compares the answer with the excerpts it was written from,
  claim by claim. A claim that is true about FastAPI but is not in the excerpts
  is unsupported -- that is the hallucination retrieval is supposed to prevent,
  even when it happens to be right.

The judge is gpt-4o, a stronger model than the one answering (gpt-4o-mini).
Neither judge sees which system wrote an answer; citation markers are stripped
before judging (see generation.strip_citations).

A refusal is never sent to a judge: on an answerable question it is scored
incorrect without asking, and it makes no claims to check.

Whether these judges can be trusted is measured, not assumed -- see
calibration.py and scripts/07_validate_judge.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLMClient

JUDGE_MODEL = "gpt-4o"
VERDICTS = ("correct", "partial", "incorrect")

CORRECTNESS_SYSTEM = """\
You grade answers to questions about the FastAPI web framework. You are given a question, a reference answer taken from FastAPI's documentation, and a candidate answer. Treat the reference answer as ground truth.

- correct: the candidate gives the reference's key facts, so that a user acting on it would get it right, and says nothing that contradicts the reference.
- partial: nothing in the candidate contradicts the reference, but it leaves out a key fact the question needs.
- incorrect: the candidate contradicts the reference on any point, gets the main point wrong, answers a different question, or does not answer.

Wording, length, order and extra detail do not matter, unless the extra detail contradicts the reference. Code in the reference illustrates the answer; the candidate need not repeat it, but must not contradict it. Judge against the reference, not against what you believe about FastAPI.

Reply with a JSON object: {"reason": "<one or two sentences>", "verdict": "correct" | "partial" | "incorrect"}"""

FAITHFULNESS_SYSTEM = """\
You check whether an answer is supported by the documentation excerpts it was written from.

Split the answer into its factual claims: each statement a user could act on or check. For each claim, decide whether the excerpts support it, meaning the excerpts state it or directly imply it. A claim that is true about FastAPI but is not in the excerpts is NOT supported. Ignore connecting phrases, and statements that only say what the excerpts do not cover.

Reply with a JSON object: {"claims": [{"claim": "<the claim, in a few words>", "supported": true | false}]}"""


def correctness_prompt(question: str, reference: str, answer: str) -> str:
    return (f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
            f"Candidate answer:\n{answer}")


def faithfulness_prompt(context: str, answer: str) -> str:
    # Excerpts first: the long part of the prompt is then a shared prefix
    # across every answer judged against the same context.
    return f"Excerpts:\n\n{context}\n\nAnswer:\n{answer}"


@dataclass
class Correctness:
    verdict: str  # one of VERDICTS, or "invalid" if the judge's reply could not be read
    reason: str = ""

    @property
    def correct(self) -> bool:
        return self.verdict == "correct"


@dataclass
class Faithfulness:
    claims: list[dict] = field(default_factory=list)  # [{"claim": str, "supported": bool}]
    valid: bool = True

    @property
    def unsupported(self) -> list[str]:
        return [c["claim"] for c in self.claims if not c["supported"]]

    @property
    def faithful(self) -> bool:
        """Every claim supported. An answer with no claims at all is not
        counted as faithful -- there is nothing it was faithful about."""
        return self.valid and bool(self.claims) and not self.unsupported


def parse_correctness(text: str) -> Correctness:
    try:
        obj = json.loads(text)
        verdict = str(obj.get("verdict", "")).strip().lower()
        reason = str(obj.get("reason", ""))
    except (json.JSONDecodeError, AttributeError):
        return Correctness("invalid", text[:200])
    return Correctness(verdict if verdict in VERDICTS else "invalid", reason)


def parse_faithfulness(text: str) -> Faithfulness:
    try:
        raw = json.loads(text)["claims"]
        claims = [{"claim": str(c["claim"]), "supported": c["supported"]} for c in raw]
    except (json.JSONDecodeError, KeyError, TypeError):
        return Faithfulness(valid=False)
    # A judge that answers "yes" or 1 instead of true has not followed the
    # format; count the reply as unreadable rather than guess.
    if not all(isinstance(c["supported"], bool) for c in claims):
        return Faithfulness(valid=False)
    return Faithfulness(claims)


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def correctness(self, question: str, reference: str, answer: str) -> Correctness:
        resp = self.llm.complete(correctness_prompt(question, reference, answer),
                                 system=CORRECTNESS_SYSTEM, max_tokens=300, json_mode=True)
        return parse_correctness(resp.text)

    def faithfulness(self, context: str, answer: str) -> Faithfulness:
        resp = self.llm.complete(faithfulness_prompt(context, answer),
                                 system=FAITHFULNESS_SYSTEM, max_tokens=1000, json_mode=True)
        return parse_faithfulness(resp.text)
