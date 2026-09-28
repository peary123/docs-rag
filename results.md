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
