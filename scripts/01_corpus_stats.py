"""Describe the corpus: what is indexed, what is left out and why, and how big.

Usage:
    python scripts/01_corpus_stats.py

Every page is rendered, including the excluded ones, so an include that no
longer resolves fails here -- not halfway through building an index.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config, corpus  # noqa: E402


def main() -> None:
    config.require_data()
    pages = [corpus.load_page(doc_id) for doc_id in corpus.page_ids()]
    included = [p for p in pages if corpus.exclusion_reason(p.doc_id) is None]
    excluded: dict[str, list[corpus.Document]] = defaultdict(list)
    for p in pages:
        reason = corpus.exclusion_reason(p.doc_id)
        if reason:
            excluded[reason].append(p)

    print(f"FastAPI {config.FASTAPI_VERSION} @ {config.FASTAPI_COMMIT[:10]}, docs/en/docs\n")
    print(f"pages found     {len(pages):>8}")
    print(f"  excluded      {len(pages) - len(included):>8}")
    for reason, docs in excluded.items():
        size = sum(len(d.text) for d in docs)
        print(f"    {len(docs):>3} page(s), {size:>9,} chars  {reason}")
    print(f"  included      {len(included):>8}\n")

    chars = sum(len(d.text) for d in included)
    words = sum(len(d.text.split()) for d in included)
    inc_chars = sum(d.included_chars for d in included)
    n_inc = sum(d.includes for d in included)
    n_inc_pages = sum(1 for d in included if d.includes)
    headings = [h for d in included for h in d.headings]
    sizes = sorted(len(d.text) for d in included)

    print("included pages")
    print(f"  characters    {chars:>10,}")
    print(f"  words         {words:>10,}")
    print(
        f"  from includes {inc_chars:>10,}  ({100 * inc_chars / chars:.1f}% of characters; "
        f"{n_inc} includes on {n_inc_pages} pages)"
    )
    print(f"  headings      {len(headings):>10,}")
    print(
        f"  page chars    min {sizes[0]:,} / median {int(statistics.median(sizes)):,} / "
        f"p90 {sizes[int(0.9 * (len(sizes) - 1))]:,} / max {sizes[-1]:,}"
    )
    by_size = sorted(included, key=lambda d: len(d.text))
    print("  largest       " + ", ".join(f"{d.doc_id} ({len(d.text):,})" for d in by_size[:-6:-1]))
    print("  smallest      " + ", ".join(f"{d.doc_id} ({len(d.text):,})" for d in by_size[:5]))

    notes = next(p for p in pages if p.doc_id == "release-notes.md")
    others = sum(len(p.text) for p in pages if p.doc_id != "release-notes.md")
    print(
        f"\nrelease-notes.md alone: {len(notes.text):,} chars; "
        f"every other page together: {others:,} chars"
    )


if __name__ == "__main__":
    main()
