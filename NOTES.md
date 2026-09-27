# Notes

Working notes: decisions I had to make and things that cost me time. Kept
because in six months I will not remember why any of this is the way it is.

---

## Decisions

### FastAPI's docs, pinned to one commit

The corpus had to be public, markdown in the repository (a PDF corpus turns
the project into a PDF-parsing project), and a few hundred pages at most. The
FastAPI docs fit: 155 English pages, written as markdown next to the code.

They also have the property the retrieval comparison needs. Questions about an
API are full of exact identifiers — `Depends`, `response_model`,
`HTTPException`, `--proxy-headers` — which is where keyword matching and
embeddings should disagree most. A corpus of plain prose would make BM25 vs
dense a less interesting fight.

The release is pinned by commit, not tag, and not "latest". A question set is
written against one version of the docs, and FastAPI moves fast — 22 releases
in the six weeks up to 0.141.1. An answer that is right for one release can be
wrong for the next, and the question set would rot silently.

**The known weakness, stated up front.** gpt-4o-mini's training data runs to
October 2023, so it has read an older version of these docs and can answer
some questions with no retrieval at all. That makes "the answer was correct"
weak evidence that retrieval worked. The control is to measure it: answer the
question set closed-book too, so the value of retrieval is a measured
difference rather than an assumption. The docs have moved since the cutoff —
they now teach `fastapi dev` and recommend `uv`, and cover FastAPI Cloud —
which is where closed-book answers should break.

### The markdown is not the page

31.3% of the characters in the index are not in the markdown files. 446
include directives on 90 of the 121 indexed pages pull code in from
`docs_src/` when the site is built:

```
{* ../../docs_src/query_params/tutorial002_py310.py hl[7] *}
```

Index the raw files and the page that answers "how do I make a query parameter
optional?" has had its example replaced by a file path.

So `src/corpus.py` does the part of the site build that changes which words are
on the page, and nothing else:

- **Includes** are resolved, with `ln[...]` line ranges (1-based, inclusive,
  comma-separated). Excerpts get the same "Code above omitted 👆" / "Code here
  omitted 👈" / "Code below omitted 👇" lines the live site prints — copied
  from the rendered page — so the model can tell an excerpt from a whole file.
- **Admonitions and tabs** (`/// tip`, `//// tab | Python 3.10+`) become a
  one-line bold label. The marker syntax is noise to BM25 and to an embedding.
- **Heading anchors** (`{ #defaults }`) move out of the text into metadata.
- **HTML** becomes its text. `<dfn title="...">` keeps its tooltip in brackets,
  because the tooltip is often the only plain-English gloss of a term. Inside
  code blocks HTML is left alone — in a Jinja example it *is* the code —
  except in `console` blocks, where it is terminal colouring for the site's
  animated terminal.

Not reproduced: the collapsed "Full file preview" the site adds under an
excerpt. It repeats the whole file, and near-duplicate chunks crowd each other
out of a top-k.

### What is left out

The rule is "pages that teach how to use FastAPI". 34 of 155 pages fail it:

- **`release-notes.md`** — a changelog. It would also be 31% of the index on its
  own (398,580 of 1,295,750 characters), so every retrieval would be competing
  with version bump entries.
- **`reference/`, 24 pages** — the API reference is generated from docstrings at
  build time. The markdown holds a sentence and a `::: fastapi.BackgroundTasks`
  stub. Reconstructing it from the library source is its own project, so
  questions about exact parameter lists are out of scope for the question set.
- **9 pages about the project** — its people, translations, repository
  management, a newsletter sign-up, and a test fixture for the translation
  tooling.

The three section landing pages are kept even though they are tiny
(`about/index.md` is 59 characters). They are on the site; whether a
near-empty chunk ever wins a retrieval is something to look at when there is
retrieval to look at.

### An include that cannot be resolved is an error

A page with a skipped include still renders and still indexes. It is just
missing its example, and nothing downstream would ever notice. So a missing
file, a line range past the end of a file, or a path that leaves the corpus
directory raises.

That paid for itself on the first run: `how-to/configure-swagger-ui.md`
includes from the library itself (`../../fastapi/openapi/docs.py`), not from
`docs_src/`. The data script now extracts the `fastapi/` package too.

### Every citation can link to its section

All 1,080 headings on the indexed pages carry an explicit `{ #anchor }`, and
each heading's character offset in the rendered text is kept. So a passage
from anywhere on a page — whichever way the page ends up being chunked — can
be cited as the URL of the section it starts in, e.g.
`https://fastapi.tiangolo.com/tutorial/query-params/#defaults`. A fallback
generates the site's own slug for a heading without an anchor; no indexed page
needs it.

### Hash the files, not the archive

GitHub generates source archives on the fly and does not promise their bytes
stay the same — in 2023 a change to its compression broke checksums across
package managers. The files inside a commit cannot change, so
`00_prepare_data.py` checks a digest of the extracted files instead.
Downloading the same release by tag and by commit gives the same digest.

### An answer's location is a span of text, not a chunk id

The obvious way to record where a question's answer lives is the id of the
chunk it came from. It does not work here: chunking is compared across two
strategies, and a chunk id from one does not exist in the other, so a question
set keyed on one chunking cannot score the other.

So the gold location is a character span in the rendered page. A retrieved
chunk is a hit when it overlaps the span, whichever chunking produced it. Each
span also stores the exact text it covers; `validate` re-reads the page at those
offsets, so a renderer change that shifts the text fails loudly instead of
quietly scoring every question against the wrong words.

### The questions are written by a stronger model than the one that answers

gpt-4o writes the candidates; gpt-4o-mini will answer them. Models tend to rate
answers phrased the way they would phrase them as better, and a question set
written by the answering model carries its phrasing into the reference answers.
Using a different model for each role is the cheap half of the fix. The
expensive half, which is not optional, is that a person reviews every question.

### Questions in the user's words, and the number that shows it

A model writing a question from a passage borrows the passage's words — "How do
I declare optional query parameters by setting their default?" — and a question
set made of those rewards keyword search for matching the docs to themselves.
BM25 against dense retrieval is one of the comparisons this project exists for,
so that bias would land directly in a headline number.

The generator is told to ask as someone who has not read the page, and every
candidate records the longest run of consecutive words it shares with its
source. Of the 124 offered for review, 99 share at most three words and 12
share five or more. The review page flags those 12, and the build step reports
the figure again for the final set.

### Review is done by a person, and recorded

Generation is automated; judgement is not. Each candidate is kept, edited, or
dropped with a reason, in a local page that shows it next to its source text.
The reviewer also writes the questions the docs *cannot* answer — those test
whether the system admits it does not know, and only a person can check that a
plausible question really has no answer here. The page has a search box for
exactly that check.

Every decision is saved with a timestamp to `eval/review_state.json`, so the
file doubles as the record of how the set was made: how many were kept as
generated, how many were edited, how many dropped, and for what.

---

## Things that bit me

### Code comments look like headings

138 lines inside the indexed pages' code blocks start with `# ` — Python
comments like `# This is not asynchronous`. There are 1,080 real headings. A
section splitter that doesn't track code fences would invent 138 sections, and
put a heading in the middle of a code example. The renderer tracks fences for
everything, and a test pins it.

### The docs show the wrong code on one page

`how-to/configure-swagger-ui.md` says "It includes these default
configurations:" and then includes lines 9–24 of `fastapi/openapi/docs.py`.
Those lines were the defaults until a February 2026 refactor
(fastapi/fastapi#14986) added a `_html_safe_json` helper above them, which
pushed the whole declaration down 14 lines, from 8–23 to 22–37. The line range
was never updated. The live page (checked 2026-09-24) shows the helper function where
the defaults should be; they only appear in the collapsed full-file preview.
It is still wrong on master.

The index follows the page, so the defaults are not in it. That is the right
behaviour for a system that answers from the docs — and it means the question
set must not ask for them.

### A size I wrote down before measuring it

The first version of the changelog's exclusion reason said it was "larger than
all other pages combined". True of the files — 696 KB against 771 KB — and
false of what gets indexed: 398,580 characters against 933,645. Most of a
changelog's bytes are link markup (every entry links a PR and an author), and
the other pages gain 281k characters of included code. Measure the rendered
text, not the files.

### Asking for JSON changed the quotes I asked to be copied

The generator returns JSON and must copy its evidence word for word. The first
run rejected 39 of 140 candidates because a quote could not be found in its
source. That looked like the model paraphrasing. It was mostly my harness:

    source: Python will complain if you put a value with a "default" before...
    quote:  Python will complain if you put a value with a 'default' before...

Writing a JSON string, the model swaps `"` for `'` so it does not have to escape
them. It also drops the `**` around bold words. The words were intact every time.

The matcher now ignores quote-mark style, backticks, `*` emphasis and
whitespace, and still requires every word, in order. Re-running from the cache
(no API calls) took the rejections from 39 to 12 and the candidates offered for
review from 97 to 124. The 12 left are genuine changes — a `#` dropped from a
code comment, a comma turned into a full stop, text merged across a code
fence — and stay rejected.

### A browser and Python disagree about where a character is

The review page lets the reviewer select evidence, and sends back the
selection's offsets. JavaScript counts a string in UTF-16 units; Python counts
code points. They agree until an emoji — and this corpus has them, in every
"Code above omitted 👆" marker the renderer inserts. Every selection after one
would have been off by a character.

The page converts its offsets to code points before sending them, and the server
does not trust them anyway: it checks that the text at those offsets is the text
the reviewer selected, and otherwise searches the section for it. Tested by
selecting text after an emoji on three different pages.
