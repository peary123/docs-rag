"""Filesystem layout and the pinned corpus version.

Everything that needs to know *where* things live, or *which* release of the
FastAPI docs is being indexed, asks this module.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Read KEY=value lines from a .env file into the environment.

    Ten lines instead of a dependency. Variables already set in the environment
    win, so an explicitly exported key is never silently overridden by a stale
    file. `.env` is gitignored -- an API key must never reach the repository.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Done at import time, before anything below reads os.environ.
_load_dotenv(PROJECT_ROOT / ".env")

# The docs change every week. The question set is written against one release,
# and an answer that was right for 0.141.1 can be wrong for the next one, so the
# release is pinned -- by commit, because a tag can be moved and a commit
# cannot.
FASTAPI_VERSION = "0.141.1"
FASTAPI_COMMIT = "95f8322ee1dcda7ceace7b1c4f6c9915b36d748f"
SITE_URL = "https://fastapi.tiangolo.com/"

DATA_DIR = Path(os.environ.get("DOCS_RAG_DATA_DIR", PROJECT_ROOT / "data"))
CORPUS_DIR = DATA_DIR / f"fastapi-{FASTAPI_VERSION}"

# Include directives in the pages are written relative to docs/en/, because
# that is the directory the docs build runs from: `../../docs_src/x.py` from
# any page, however deeply nested, means <repo>/docs_src/x.py.
DOCS_BUILD_DIR = CORPUS_DIR / "docs" / "en"
DOCS_DIR = DOCS_BUILD_DIR / "docs"
DOCS_SRC_DIR = CORPUS_DIR / "docs_src"

CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", PROJECT_ROOT / ".llm_cache"))
RESULTS_DIR = PROJECT_ROOT / "results"


def require_data() -> None:
    """Fail loudly and usefully if the corpus has not been downloaded."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"FastAPI docs not found at {DOCS_DIR}.\n"
            "Run:  python scripts/00_prepare_data.py"
        )
