# Results

Every experiment, with the configuration that produced it. Per-question records
are in `results/`; `python scripts/05_retrieval_eval.py` reproduces the
retrieval numbers exactly (embeddings are cached, and nothing in it is random).

---

## Retrieval

The 116 answerable questions from `eval/questions.jsonl`, against two chunkings
and four retrievers. The 12 unanswerable questions have no evidence to find and
are left out of retrieval scoring; they matter for generation.

**What counts as a hit.** A retrieved chunk is a hit when it covers at least
half of one of the question's evidence spans. Every evidence span is shorter
than a typical chunk under either scheme (median 101 characters, longest 413),
so the rule is reachable under both and favours neither. Under the loosest rule
— any overlap at all — every recall@5 below moves by at most 1.8 points and no
conclusion changes.

**Settings, all fixed before the first run and none tuned on the questions:**
500-token chunks with 50 overlap, counted with the embedding model's own
tokenizer; `BAAI/bge-small-en-v1.5` with its recommended query prefix;
Reciprocal Rank Fusion with k = 60 over the top 50 of each; the cross-encoder
`ms-marco-MiniLM-L-6-v2` reordering the hybrid top 20.

| chunking | retriever | recall@5 | 95% CI | MRR@10 | all evidence in top 5 | p50 latency |
|---|---|---|---|---|---|---|
| fixed (613 chunks) | BM25 | 81.0% | 73–87% | 0.635 | 68.1% | 1 ms |
| | dense | 81.0% | 73–87% | 0.677 | 67.2% | 10 ms |
| | hybrid | 82.8% | 75–89% | 0.705 | 70.7% | 12 ms |
| | hybrid + rerank | **86.2%** | 79–91% | 0.709 | 71.6% | 580 ms |
| headings (1,204 chunks) | BM25 | 66.4% | 57–74% | 0.513 | 60.3% | 2 ms |
| | dense | 73.3% | 65–81% | 0.562 | 69.0% | 10 ms |
| | hybrid | 78.4% | 70–85% | 0.603 | 72.4% | 14 ms |
| | hybrid + rerank | 80.2% | 72–86% | 0.606 | 73.3% | 693 ms |

Latency is per query on a laptop CPU, including encoding the query; building
the indexes is excluded.

### The comparisons decided before the run

Paired over the same 116 questions: McNemar's exact test on recall@5 (only the
questions two configurations disagree on carry information), and a paired
bootstrap interval on MRR.

| comparison | recall@5 | fixed / broken | McNemar p | MRR change, 95% CI |
|---|---|---|---|---|
| fixed: BM25 → hybrid | +1.7 | 8 / 6 | 0.79 | **+0.070** [+0.008, +0.132] |
| fixed: dense → hybrid | +1.7 | 5 / 3 | 0.73 | +0.028 [−0.031, +0.085] |
| fixed: hybrid → + rerank | +3.4 | 10 / 6 | 0.45 | +0.004 [−0.049, +0.057] |
| headings: BM25 → hybrid | **+12.1** | 17 / 3 | **0.003** | **+0.090** [+0.039, +0.140] |
| headings: dense → hybrid | +5.2 | 13 / 7 | 0.26 | +0.041 [−0.005, +0.087] |
| headings: hybrid → + rerank | +1.7 | 5 / 3 | 0.73 | +0.003 [−0.053, +0.059] |
| BM25: fixed → headings | **−14.7** | 3 / 20 | **<0.001** | **−0.122** [−0.186, −0.056] |
| dense: fixed → headings | −7.8 | 9 / 18 | 0.12 | **−0.115** [−0.193, −0.033] |
| hybrid: fixed → headings | −4.3 | 8 / 13 | 0.38 | **−0.102** [−0.177, −0.025] |
| hybrid + rerank: fixed → headings | −6.0 | 6 / 13 | 0.17 | **−0.103** [−0.166, −0.039] |

Read alone, the last four rows say fixed-length chunks beat heading chunks with
every retriever. They don't — see below.

### At an equal context budget, the chunking gap disappears

*This check was added after the run above, when it became clear the comparison
was unequal; it was not planned in advance.*

A fixed chunk is ~500 tokens and a heading chunk a median ~180, so "top 5"
gives the fixed scheme nearly three times as much text. More text covers more
evidence regardless of how it was cut. The fair question — and the one
generation actually faces — is what fits in the same amount of context. So
each ranking is also scored on the chunks that fit in 2,500 tokens, about five
fixed chunks or nine heading chunks:

| retriever | fixed @ 2.5k tokens | headings @ 2.5k tokens | difference | McNemar p |
|---|---|---|---|---|
| BM25 | 81.0% | 75.9% | −5.2 | 0.26 |
| dense | 81.0% | 81.9% | +0.9 | 1.00 |
| hybrid | 82.8% | **86.2%** | +3.4 | 0.45 |
| hybrid + rerank | **86.2%** | 84.5% | −1.7 | 0.82 |

**None of the differences survives.** The 4- to 15-point gaps at equal k were
chunk size, not chunk boundaries. The MRR gaps in the table above were not
re-tested this way — MRR has no natural equal-budget form — and are probably
the same effect: a list of smaller pieces puts the first hit further down.

### What the numbers support

- **Hybrid is never worse than either retriever alone, and clearly better than
  BM25 alone on heading chunks** (+12.1 recall@5, p = 0.003). Its edge over
  dense alone is positive under both chunkings but within the noise.
- **The reranker adds nothing measurable here, at 50 times the latency.** +3.4
  and +1.7 recall@5, neither significant; MRR unchanged; 12 ms becomes ~600 ms.
  The cross-encoder was trained on web search passages, and these are
  documentation pages half made of code.
- **Chunking makes no measurable difference once the context budget is equal.**
- **27–40% of questions do not get all their evidence into the top 5**,
  depending on the configuration, which bounds how complete a generated answer
  can be.

### Carried into generation

**Heading chunks, hybrid retrieval, a 2,500-token context budget.** Tied for the
best at that budget (86.2%) with fixed + rerank, at 14 ms instead of 580 ms, and
every heading chunk is a whole section, which is the unit an answer cites.
Decided here, before any generation run.

---

## Generation and end-to-end evaluation

### Fixed before the first run

**System.** Heading chunks, hybrid retrieval, the excerpts that fit in 2,500
tokens (as carried out of retrieval). `gpt-4o-mini` at temperature 0 answers
from the numbered excerpts only, cites them as [n], and replies with exactly
"The documentation does not cover this." when they do not answer the question.
The prompt is in `src/generation.py`; it is not tuned on the question set, and
would be changed only if its output could not be read, with the change recorded
here.

**Closed-book control.** The same model and questions with no excerpts, told to
reply "I don't know." when it doesn't. The model was trained on an older version
of these docs; the control measures how much of the answering it can do from
memory, so the value of retrieval is a measured difference.

**Metrics.**

| metric | on | how |
|---|---|---|
| correctness | 116 answerable | `gpt-4o` judge: correct / partial / incorrect against the reference answer. Headline: share judged *correct*. A refusal is incorrect, without asking the judge. |
| faithfulness | every answer that is not a refusal | `gpt-4o` judge splits the answer into claims and checks each against the excerpts the answer was written from. Headline: share of answers with every claim supported. |
| refusal accuracy | 12 unanswerable | the answer is exactly the refusal sentence — string match, no judge |
| false refusals | 116 answerable | the same match |
| citations | answered, evidence in context | the answer cites at least one excerpt that covers the evidence — span overlap, no judge |

Judges never see which system wrote an answer; citation markers are stripped
first.

**Comparison decided in advance:** correctness with retrieval against
closed-book, paired over the 116 answerable questions (McNemar). Reported
alongside, without a test: correctness split by whether the evidence reached the
model.

**The judges are validated before their scores are used**, on answers whose
right verdict is known by construction (`src/calibration.py`): each reference
answer reworded (should be judged correct), with one fact changed (should not),
swapped for the most similar other question's answer (should not); and, against
the excerpts, the reference answer as is (should be supported) and with one
invented sentence added (should not). No answer was graded by hand.

### Can the judges be trusted?

`python scripts/07_validate_judge.py`. 498 answers whose right verdict is known
by construction, written by `gpt-4o-mini` from the reference answers and graded
by the `gpt-4o` judges with the prompts used for scoring below. Faithfulness
probes use the excerpts actually retrieved for the question, and only the 92
questions whose evidence all made it into them.

| probe | should be | built | judged as it should be | 95% CI | verdicts |
|---|---|---|---|---|---|
| reference, reworded | correct | 112 | **112 (100%)** | 96.7–100% | correct 112 |
| one fact changed | not correct | 86 | 77 (89.5%) | 81.3–94.4% | incorrect 77, correct 9 |
| another question's answer | not correct | 116 | 112 (96.6%) | 91.5–98.7% | incorrect 103, partial 9, correct 4 |
| reference, against excerpts | faithful | 92 | 91 (98.9%) | 94.1–99.8% | |
| reference + invented sentence | not faithful | 92 | 87 (94.6%) | 87.9–97.7% | |

**Planted errors rejected: 276 of 294 (93.9%, CI 90.5–96.1%); correct answers
accepted: 112 of 112 paraphrases, 91 of 92 supported answers.** Paired, the
invented sentence was caught in 86 of the 91 answers whose clean version passed.

Construction checks threw out 4 paraphrases that lost an identifier and 30
substitutions (20 named a phrase longer than 80 characters, 9 named a phrase
that does not occur exactly once, 1 incomplete reply).

**Reading the 19 disagreements**, most turn out to be the construction's
mistake, not the judge's:

| | count | probes |
|---|---|---|
| the planted change was harmless: only a name the user chooses, or still true | 4 | q038, q072, q082, q087 (substitution) |
| the "other" question is a near-duplicate, and its answer does answer this one | 4 | q030, q052, q102, q103 (swap) |
| the "invented" sentence is in the excerpts after all | 4 | q053, q091, q094, q108 (fabricated) |
| the reference answer itself goes beyond the docs | 1 | q105 (supported) |
| **judge missed a real error: a name inside code** | **5** | q055 `app.webhook` for `app.webhooks`, q060 a keyword the question itself gives, q083 `data=` for `content=`; q040 and q084 rename something in one place only, so the code no longer fits together |
| borderline | 1 | q061 (fabricated) |

The same weakness shows among the agreements: the judge also rejected some
harmless renames, such as q021 (an exception class the user names) and q073
(the user's own dependency functions). **The judge is reliable on facts stated in prose and
inconsistent on names inside code**, in both directions. The agreements were
not all read; the counts above are the construction-based numbers, uncorrected.

### End-to-end results

`python scripts/06_generate_answers.py`, then `python scripts/08_score_answers.py`.
The excerpts reaching the model hold the evidence for 86.2% of the answerable
questions, the same figure as the retrieval comparison (checked by the script).
Mean context: 2,341 tokens, 9.1 excerpts.

| | with retrieval | closed-book |
|---|---|---|
| correct, 116 answerable | **103 (88.8%)**, CI 81.8–93.3% | 70 (60.3%), CI 51.2–68.8% |
| partial / incorrect | 5 / 8 (5 of the 8 are refusals) | 12 / 34 |
| refused, 12 unanswerable | **12 of 12** | 0 of 12 |
| refused, 116 answerable | 5 (4.3%) | 1 |
| every claim supported by its excerpts | 94 of 111 answers (84.7%), CI 76.8–90.2% | — |
| cites an excerpt holding the evidence | 93 of 99 (93.9%) | — |

**Retrieval adds 28.4 points of correctness** over the same model closed-book:
39 questions go from wrong to right and 6 the other way (McNemar p < 0.001) —
the comparison decided in advance.

**Where it fails is retrieval.** Split by whether the evidence reached the
model (reported, not tested):

| | questions | correct | refused | faithful (answered) |
|---|---|---|---|---|
| evidence in the excerpts | 100 | 94 (94.0%) | 1 | 87 of 99 (87.9%) |
| evidence not in the excerpts | 16 | 9 (56.2%) | 4 | 7 of 12 (58.3%) |

7 of the 13 answers that are not correct come from the 16 questions retrieval
missed. The 6 questions closed-book gets right and retrieval does not are all
the same story: 5 had their evidence missed by retrieval, and 4 of the 6 are
refusals — the system declining rather than answering from memory, which is the
behaviour asked for.

**Refusals.** Every unanswerable question was refused. Of the 5 answerable
questions refused, 4 had no evidence in the excerpts, so refusing was the
grounded answer; one (q100) was refused with the evidence present. Closed-book,
the model refused none of the 12 — it answered all of them from memory, and
some of those answers are right (q119's `redirect_slashes=False` is a real
parameter of `FastAPI()`, documented only in pages this index leaves out). The closed-book
number shows that the model never declines, not that its 12 answers are wrong.

**Faithfulness.** 19 of 431 claims (4.4%) were judged unsupported, spread over
17 answers. *Post hoc, on reading them:* 12 of the 17 are flagged only for a
closing line saying what "the documentation" or "the excerpts" do not cover —
counted by pattern in the script. The prompt asks the model to say what the
excerpts don't cover when they answer only part of a question; 14 answers do,
9 of them in terms of the whole documentation, which is often false (q032
says the documentation does not cover deployment processes; it has a chapter on
them). The other 5 contain substantive unsupported content: 3 on questions
whose evidence retrieval missed, where the model filled the gap from memory
(q007's `uvicorn main:app --workers 4`, q015, and q075's whole answer), plus one invented
example header (q079) and one vague line (q057).

About $3 in API calls for this section (answers, probes, judging), all cached.
