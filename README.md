# RAG over the FastAPI docs

Question answering over FastAPI's documentation, with answers that cite the
section they came from, and a hand-checked question set that every retrieval
choice is measured against.

So far this repository holds the corpus — what gets indexed, and how it is
turned into text — and the tooling that builds the question set. Retrieval and
answer generation are not in yet.

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
its source, evidence highlighted. It is kept, edited, or dropped with a reason,
and the reviewer adds questions the docs *cannot* answer, to test whether the
system admits it doesn't know. Every decision is timestamped in
`eval/review_state.json`. **The review is in progress; the final set's numbers
will be reported here when it is done.**

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
12 share five or more. The reviewer sees that number as a warning.

**The questions are written by a different model from the one that will answer
them** — gpt-4o writes, gpt-4o-mini answers — since a model tends to favour
answers phrased the way it would phrase them.

## Running it

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

## Layout

```
src/
  config.py     paths, the pinned FastAPI version, .env loading
  corpus.py     renders each page into indexed text + section anchors
  evalset.py    sampling sections, locating evidence, validating the set
  review.py     review decisions, hand-written questions, section search
  llm.py        model calls with a disk cache
scripts/        entry points, numbered in the order they're useful;
                review.html is the review page
eval/           candidates, the review record, and the question set
tests/          37 tests
NOTES.md        decisions, and what went wrong
```

---

FastAPI and its documentation are MIT-licensed, © Sebastián Ramírez.
