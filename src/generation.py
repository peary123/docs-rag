"""Answering a question from retrieved documentation, with citations.

The configuration is the one carried out of the retrieval comparison: heading
chunks, hybrid retrieval, and whatever fits in 2,500 tokens of context. The
model sees the excerpts numbered [1], [2], ... and is told to

* answer only from them -- not from what it remembers about FastAPI,
* cite the excerpt numbers it used, and
* reply with one fixed sentence when they do not contain the answer.

The fixed sentence is what makes refusal measurable without a judge: an answer
either is that sentence or it isn't. A refusal the model words its own way is
not counted as one -- the prompt asks for the exact sentence, and following
that instruction is part of what is being measured.

The closed-book control asks the same model the same question with no excerpts,
to measure how much of the answering it could do from memory alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .chunking import Chunk
from .corpus import Document
from .llm import LLMClient
from .retrieval import RERANK_DEPTH
from .scoring import BUDGET_TOKENS, within_budget

ANSWER_MODEL = "gpt-4o-mini"
ANSWER_MAX_TOKENS = 700

REFUSAL = "The documentation does not cover this."
CLOSED_BOOK_REFUSAL = "I don't know."

SYSTEM = f"""\
You answer questions about the FastAPI web framework using only the documentation excerpts you are given.

- Use only what the excerpts say. Do not add anything from your own knowledge of FastAPI, even if you believe it is true.
- After each sentence that uses an excerpt, cite it by number in square brackets, like [2]. Cite only excerpts that support that sentence.
- Answer the question directly, in a few sentences. Include a short code snippet only when the question asks how to write something and the excerpts show the code.
- If the excerpts answer only part of the question, answer that part and say what they don't cover.
- If the excerpts do not answer the question at all, reply with exactly this sentence and nothing else: {REFUSAL}"""

CLOSED_BOOK_SYSTEM = f"""\
You answer questions about the FastAPI web framework.

- Answer the question directly, in a few sentences. Include a short code snippet only when the question asks how to write something.
- If you do not know the answer, reply with exactly this sentence and nothing else: {CLOSED_BOOK_REFUSAL}"""


def excerpt_heading(chunk: Chunk, doc: Document) -> str:
    """'Page title > Section title' and the section's URL, for one excerpt."""
    section = doc.section_at(chunk.start)
    path = doc.title if section is None or section.level == 1 else f"{doc.title} > {section.title}"
    return f"{path}\n{doc.cite(chunk.start)}"


def format_context(chunks: Sequence[Chunk], docs: dict[str, Document]) -> str:
    return "\n\n".join(
        f"[{n}] {excerpt_heading(c, docs[c.doc_id])}\n{c.text.strip()}"
        for n, c in enumerate(chunks, 1))


def answer_prompt(question: str, context: str) -> str:
    return f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"


# ------------------------------------------------------------ reading answers

_WRAPPING = "\"'`* \n\t"


def is_refusal(text: str, phrase: str = REFUSAL) -> bool:
    """True when the answer is the refusal sentence and nothing else.

    Tolerates what a model wraps a sentence in -- quotes, bold, a missing final
    full stop, letter case -- but not extra words: "The documentation does not
    cover this, but..." is an answer, and is scored as one.
    """
    norm = text.strip(_WRAPPING).rstrip(".").strip(_WRAPPING).lower()
    return norm == phrase.rstrip(".").lower()


def mentions_refusal(text: str, phrase: str = REFUSAL) -> bool:
    """The refusal sentence appears somewhere in a longer answer -- a partial
    refusal. Counted separately so a prompt that produces many is visible."""
    return phrase.rstrip(".").lower() in text.lower() and not is_refusal(text, phrase)


_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
# [2], [1, 3], [1][4] -- but not items[0]: a bracket straight after a word
# character is indexing, not a citation.
_CITATION = re.compile(r"(?<!\w)\[(\d+(?:\s*,\s*\d+)*)\]")


def citations(text: str) -> list[int]:
    """Excerpt numbers the answer cites, in order of first use, code ignored."""
    prose = _CODE.sub(" ", text)
    seen: list[int] = []
    for m in _CITATION.finditer(prose):
        for n in m.group(1).split(","):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def strip_citations(text: str) -> str:
    """The answer without its [n] markers -- what a judge sees, so a judge
    cannot tell a retrieval answer from a closed-book one by its brackets."""
    pieces, last = [], 0
    for m in _CODE.finditer(text):
        pieces.append(_strip_prose(text[last:m.start()]))
        pieces.append(m.group(0))
        last = m.end()
    pieces.append(_strip_prose(text[last:]))
    return "".join(pieces).strip()


def _strip_prose(prose: str) -> str:
    out = _CITATION.sub("", prose)
    out = re.sub(r"[ \t]+([.,;:)])", r"\1", out)
    return re.sub(r"(?<=\S)[ \t]{2,}", " ", out)


# ------------------------------------------------------------------- answering

@dataclass
class Answer:
    qid: str
    mode: str  # "rag" or "closed_book"
    text: str
    refused: bool
    partial_refusal: bool
    citations: list[int] = field(default_factory=list)  # valid excerpt numbers cited
    invalid_citations: list[int] = field(default_factory=list)  # numbers with no such excerpt
    context: list[str] = field(default_factory=list)  # chunk ids, in the order shown
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def cited_chunks(self) -> list[str]:
        return [self.context[n - 1] for n in self.citations]


class Generator:
    """Retrieve within the token budget, then answer from what was retrieved."""

    def __init__(self, llm: LLMClient, retriever, chunks: Sequence[Chunk],
                 tokens: dict[str, int], docs: dict[str, Document],
                 budget: int = BUDGET_TOKENS) -> None:
        self.llm, self.retriever, self.chunks = llm, retriever, chunks
        self.tokens, self.docs, self.budget = tokens, docs, budget

    def retrieve(self, question: str) -> list[Chunk]:
        ranked = [self.chunks[i] for i in self.retriever.search(question, RERANK_DEPTH)]
        return within_budget(ranked, self.tokens, self.budget)

    def answer(self, qid: str, question: str, context: Sequence[Chunk]) -> Answer:
        resp = self.llm.complete(answer_prompt(question, format_context(context, self.docs)),
                                 system=SYSTEM, max_tokens=ANSWER_MAX_TOKENS)
        cited = citations(resp.text)
        return Answer(
            qid=qid, mode="rag", text=resp.text,
            refused=is_refusal(resp.text), partial_refusal=mentions_refusal(resp.text),
            citations=[n for n in cited if 1 <= n <= len(context)],
            invalid_citations=[n for n in cited if not 1 <= n <= len(context)],
            context=[c.chunk_id for c in context],
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            latency_s=resp.latency_s)


def closed_book(llm: LLMClient, qid: str, question: str) -> Answer:
    resp = llm.complete(f"Question: {question}", system=CLOSED_BOOK_SYSTEM,
                        max_tokens=ANSWER_MAX_TOKENS)
    return Answer(
        qid=qid, mode="closed_book", text=resp.text,
        refused=is_refusal(resp.text, CLOSED_BOOK_REFUSAL),
        partial_refusal=mentions_refusal(resp.text, CLOSED_BOOK_REFUSAL),
        input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
        latency_s=resp.latency_s)
