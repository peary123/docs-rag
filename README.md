# RAG over the FastAPI docs

Question answering over FastAPI's documentation, with answers that cite the
section they came from, and a hand-checked question set that every retrieval
choice is measured against.

So far this repository holds the corpus, the question set, a measured
comparison of two chunkings and four retrievers, and answer generation scored
by an LLM judge that is itself tested first. Serving it is not in yet.

## The corpus

FastAPI 0.141.1's English documentation, pinned by commit: **121 pages, 897k
characters**. Every heading on those pages carries an explicit anchor (1,080 of
them), so any passage can be cited as a link to its section on
fastapi.tiangolo.com.

**The markdown files are not the pages.** 31% of the indexed text isn't in
them: 446 include directives on 90 pages pull code in from `docs_src/` when the
site is built, so indexing the raw files would drop most of the examples —
and the examples are what people ask about. `src/corpus.py` renders each page
the way the site does: includes resolved, including the site's "Code above
omitted 👆" markers on excerpts; admonitions reduced to their label; HTML
reduced to its text, except inside code, where HTML *is* the code. An include
that cannot be resolved is an error, not a gap.

34 of the 155 pages are left out, each for a stated reason:

| left out | why |
|---|---|
| `release-notes.md` | a changelog, not documentation — and it alone would be 31% of the index |
| `reference/` (24 pages) | generated from docstrings at build time; the markdown holds only `::: fastapi.X` stubs |
| 9 more pages | about the project rather than about using FastAPI: its people, external links, translations, repository management, contributing, newsletter |

`python scripts/01_corpus_stats.py` prints the full breakdown.

## The question set

Every retrieval and generation choice is measured against a fixed set of
questions, so the set is built with more care than anything that uses it.

**128 questions: 116 the docs answer, 12 they cannot.** Every answerable
question carries a reference answer and the exact passages that support it,
and 114 of the 116 were changed in review.

| | questions |
|---|---|
| answerable from one section | 108 |
| answerable only from two sections | 8 |
| not answerable from the docs | 12 |
| pages the answers are drawn from | 98 of 121 |

**Candidates are generated, then every one is reviewed by hand.** A model reads
a sampled section and writes one question a user might ask about it, with a
reference answer and verbatim quotes as evidence. Sections are sampled across
pages — one per page before any page gets a second — so the set covers the
docs rather than the docs' longest pages. Twenty candidates are drawn from pairs
of nearby sections and must need both to answer.

| | generated | offered for review | rejected automatically |
|---|---|---|---|
| one section | 120 | 108 | 8 evidence not verbatim, 4 skipped by the model |
| two sections | 20 | 16 | 4 evidence not verbatim |

A candidate reaches review only if every quote it gives can be found in the
source section, word for word. The 124 offered cover 103 of the 121 pages.

Review happens in a local page (`scripts/03_review.py`): each candidate next to
its source, evidence highlighted, kept, edited, or dropped with a reason. Every
decision is timestamped in `eval/review_state.json`.

| review outcome | candidates |
|---|---|
| kept as generated | 2 |
| kept after editing | 114 |
| dropped: answer not supported by the passage | 6 |
| dropped: ambiguous | 1 |
| dropped: duplicate | 1 |

Review changed the set more than it pruned it. 90 questions were rewritten the
way a user would ask them — *"How can I serialize bytes as base64 in a FastAPI
JSON response using Pydantic?"* became *"My FastAPI response model has a bytes
field. How do I make it come out as base64 in the JSON response?"* — and 106
reference answers were expanded, 42 of them gaining the code that does it. Six
of the eight drops were answers that claimed more than their passage said: the
generator overreaches, which is the case for reviewing every candidate rather
than a sample.

**The 12 unanswerable questions test whether the system admits it doesn't
know.** They are written for the set rather than drawn from a section, and each
is a different kind of trap: a feature FastAPI doesn't have (*built-in rate
limiting*), something documented only in the unindexed API reference, a
neighbouring page that looks relevant but doesn't answer (the JWT tutorial for
*refresh tokens*), or an integration the docs never mention. Each lists the
terms any answer would have to use, and the build fails if one of them appears
anywhere in the indexed pages — so "unanswerable" is checked, not asserted.

**Where an answer lives is stored as a span of text, not a chunk id.** Two
chunking strategies will be compared, and a chunk id from one does not exist in
the other. A span in the rendered page works for both: a retrieved chunk is a
hit when it overlaps one. Each span keeps the exact text it covers, so a change
to the renderer that moves it is caught instead of silently mis-scoring.

**Questions are asked the way a user would, and that is measured.** A model
writing questions from a passage tends to reuse its wording, which flatters
keyword search for reasons that have nothing to do with real users. The
generator is told not to, and each candidate records the longest phrase it
shares with its source: 99 of 124 share no more than three consecutive words,
12 share five or more. The reviewer sees that number as a warning, and after
review **4 of the 116** answerable questions share five or more.

Only 8 questions need two sections, too few to report as their own category.
They count in every overall
figure but get no headline of their own. One question moves an 8-question
figure by 12.5 points.

**The questions are written by a different model from the one that will answer
them** — gpt-4o writes, gpt-4o-mini answers — since a model tends to favour
answers phrased the way it would phrase them.

## Retrieval

Two chunkings — fixed 500-token windows, and one chunk per section — against
four retrievers: BM25, dense (`bge-small-en-v1.5`), a hybrid of the two merged
by Reciprocal Rank Fusion, and the hybrid reranked by a cross-encoder. Scored
on the 116 answerable questions; a chunk is a hit when it covers at least half
of an evidence span.

| retriever | fixed, recall@5 | headings, recall@5 | fixed, @2.5k tokens | headings, @2.5k tokens | latency |
|---|---|---|---|---|---|
| BM25 | 81.0% | 66.4% | 81.0% | 75.9% | 1–2 ms |
| dense | 81.0% | 73.3% | 81.0% | 81.9% | 10 ms |
| hybrid | 82.8% | 78.4% | 82.8% | **86.2%** | 12–14 ms |
| hybrid + rerank | **86.2%** | 80.2% | **86.2%** | 84.5% | 580–690 ms |

**The chunking result reverses once the comparison is fair.** At an equal
number of chunks, fixed windows beat heading chunks with every retriever, by up
to 15 points. But a fixed chunk holds ~500 tokens and a heading chunk ~180, so
the top five of one is nearly three times the text of the other. At an equal
2,500-token budget — what generation will actually get — no difference between
the chunkings is significant (McNemar p ≥ 0.26). The gap was size, not
boundaries. This check was added after the first run exposed the imbalance.

**The reranker buys nothing measurable, at 50 times the latency**: +3.4 and
+1.7 recall@5, neither significant, MRR unchanged. **Hybrid is never worse than
either retriever alone** and clearly beats BM25 alone on heading chunks (+12.1
recall@5, p = 0.003).

Generation uses **heading chunks with hybrid retrieval, 2,500 tokens of
context**: tied for best at that budget, at 14 ms rather than 580, and each
chunk is a whole section — the unit an answer cites. Full tables, confidence
intervals and every paired test are in [results.md](results.md).

## Answers

`gpt-4o-mini` answers from the retrieved excerpts only, cites them by number,
and replies *"The documentation does not cover this."* when they don't answer
the question. A `gpt-4o` judge scores the answers — after the judge itself has
been tested.

| | with retrieval | same model, closed-book |
|---|---|---|
| correct, 116 answerable questions | **88.8%** | 60.3% |
| refused, 12 unanswerable questions | **12 of 12** | 0 of 12 |
| refused, 116 answerable questions | 5 | 1 |
| every claim supported by the excerpts | 84.7% | — |
| cites the excerpt that holds the evidence | 93.9% | — |

**Retrieval is worth 28 points of correctness**: 39 questions fixed, 6 broken
(McNemar p < 0.001). The model was trained on an older version of these docs,
so closed-book is the honest baseline — and it gets 60% right from memory.

**What is left is mostly retrieval's.** When the evidence reaches the model,
94% of answers are correct; for the 16 questions where it doesn't, 56%. 7 of the
13 answers that aren't correct come from those 16. Of the 6 questions
closed-book gets right and retrieval doesn't, 5 are retrieval misses and 4 end
in a refusal: the system declining rather than guessing, as it was told to.

**A refusal is a fixed sentence, matched as a string**, so refusal accuracy
needs no judge. Every unanswerable question was refused. Closed-book, the model
declined none of them.

**Most faithfulness failures come from one line of the prompt.** 12 of the 17
answers with an unsupported claim are flagged only for a closing remark about
what isn't covered. The prompt asks the model to say what *the excerpts* don't
cover; 9 of the 14 answers that do say *the documentation* instead, which is
often false — the docs cover plenty the excerpts didn't include. 5 answers
hold substantive unsupported content, 3 of them after retrieval missed the
evidence and the model filled the gap from memory. The prompt was fixed before
the run and is not tuned after it; the fix is left for the next version.

### Is the judge right?

No answer was graded by hand. Instead the judges are tested against answers
whose right verdict is known by construction. Each reference answer is
reworded (should be judged correct), has one fact changed, or is swapped for
the most similar other question's answer (should not be); and, against the
retrieved excerpts, is checked as is (supported) and with one invented sentence
added (not supported). 498 probes in all.

| probe | judged as it should be |
|---|---|
| correct answers, reworded | **112 of 112** |
| planted errors: a changed fact, a swapped answer, an invented sentence | **276 of 294 (93.9%)** |
| supported answers | 91 of 92 |

Reading the 19 disagreements, 13 are the construction's fault rather than the
judge's: a "wrong" substitution that only renamed a variable the user chooses,
a near-duplicate question whose answer does fit, an "invented" sentence that the
excerpts turn out to contain. 5 are real misses, all the same kind: a name
inside code that is wrong or no longer fits (`app.webhook` for
`app.webhooks`). **The judge is reliable on facts stated in prose and
inconsistent on names inside code** — worth knowing in a docs QA system, where
the names are half the answer.

This is weaker than hand-grading: a planted error is one kind of error, real
ones can be subtler, and a planted "error" can turn out harmless, as 13 did. What
it offers instead is scale and a trail: 498 probes rather than 30 answers, each
recording exactly what was changed, so every disagreement can be traced. Details
in [results.md](results.md#can-the-judges-be-trusted).

## Running it

Everything runs in a project virtual environment; see
[NOTES.md](NOTES.md#pytorch-and-anacondas-numpy-cannot-share-a-process) for
why not the Anaconda base environment. On Windows:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Then run every command below with `.venv\Scripts\python` in place of `python`.

```bash
python scripts/00_prepare_data.py
```

Downloads the FastAPI source archive for the pinned commit (~18 MB) into
`data/`, extracts the English docs, `docs_src/` and the library, and checks
them against a content digest. `data/` is gitignored.

```bash
python scripts/01_corpus_stats.py
```

```bash
python -m pytest tests/ -q
```

The tests that need the corpus skip cleanly without it.

Building the question set needs an OpenAI key, in a `.env` file at the repo
root (gitignored) or exported:

```bash
python scripts/02_generate_candidates.py
```

```bash
python scripts/03_review.py
```

```bash
python scripts/04_build_questions.py
```

Generation costs about $0.40 and is cached, so re-running it is free. Review
opens in the browser and saves as it goes; the build step validates every
evidence span against the corpus and writes `eval/questions.jsonl`.

```bash
python scripts/05_retrieval_eval.py
```

The retrieval comparison runs locally and costs nothing. The first run
downloads two small models and encodes both chunkings (about a minute each on a
laptop CPU); after that the vectors are cached in `.index/` and a full run takes
about three minutes, almost all of it the reranker.

```bash
python scripts/06_generate_answers.py
```

```bash
python scripts/07_validate_judge.py
```

```bash
python scripts/08_score_answers.py
```

Answers, then the judge's test, then the scores. About $3 in API calls, nearly
all of it the `gpt-4o` judge, and cached like everything else.

## Layout

```
src/
  config.py     paths, the pinned FastAPI version, .env loading
  corpus.py     renders each page into indexed text + section anchors
  evalset.py    sampling sections, locating evidence, validating the set
  review.py     review decisions, hand-written questions, section search
  chunking.py   fixed-window and per-section chunking
  retrieval.py  BM25, dense, hybrid (RRF), cross-encoder reranking
  scoring.py    hits against evidence spans, recall@5, MRR, paired tests
  generation.py answering from excerpts, citations, refusals, closed-book
  judge.py      correctness and faithfulness judges
  calibration.py answers with known verdicts, for testing the judges
  llm.py        model calls with a disk cache
scripts/        entry points, numbered in the order they're useful;
                review.html is the review page
eval/           candidates, the review record, the question set, judge probes
results/        per-question records and summaries: retrieval, answers,
                judge verdicts
tests/          63 tests
results.md      every experiment and its configuration
NOTES.md        decisions, and what went wrong
```

---

FastAPI and its documentation are MIT-licensed, © Sebastián Ramírez.
