"""Download the pinned FastAPI release and extract its English docs into data/.

Usage:
    python scripts/00_prepare_data.py

Three parts of the repository are kept: the English pages
(docs/en/docs/**/*.md); docs_src/, which holds the code the pages include at
build time; and the fastapi/ package itself, because one page includes a
default configuration straight from the library source. The archive (~18 MB)
is left in place so a re-run is a no-op.

Source: https://github.com/fastapi/fastapi (MIT License).
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config  # noqa: E402

ARCHIVE_URL = f"https://github.com/fastapi/fastapi/archive/{config.FASTAPI_COMMIT}.tar.gz"
ARCHIVE = config.DATA_DIR / f"fastapi-{config.FASTAPI_VERSION}.tar.gz"

# A digest of the extracted files, not of the archive. GitHub generates these
# archives on the fly and does not promise the compressed bytes stay identical
# (it changed them once, in 2023, and broke checksums across package managers).
# The files inside a commit cannot change, so that is what gets checked.
EXPECTED_DIGEST = "63fae407b043c329eb53df180a963f18a287aac21278798cb499bae93dea0f09"


def _wanted(rel: PurePosixPath) -> bool:
    parts = rel.parts
    if rel.name == "LICENSE" and len(parts) == 1:
        return True
    if parts[:3] == ("docs", "en", "docs") and rel.suffix == ".md":
        return True
    return parts[:1] in {("docs_src",), ("fastapi",)}


def download() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        print(f"archive already present: {ARCHIVE}")
        return
    print(f"downloading {ARCHIVE_URL}")
    tmp = ARCHIVE.with_suffix(".part")
    urllib.request.urlretrieve(ARCHIVE_URL, tmp)
    tmp.replace(ARCHIVE)  # a killed download must not look like a finished one
    print(f"  {ARCHIVE.stat().st_size / 1e6:.1f} MB")


def extract() -> int:
    """Write the wanted members under CORPUS_DIR; return how many were written.

    Members are read and written one by one rather than with extractall, so
    the destination path is always built here from a checked relative path --
    an archive entry like `../../x` cannot land outside data/.
    """
    written = 0
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            # Every entry sits under one top-level directory, fastapi-<commit>/.
            rel = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if ".." in rel.parts or not rel.parts or not _wanted(rel):
                continue
            dest = config.CORPUS_DIR.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            assert src is not None
            dest.write_bytes(src.read())
            written += 1
    return written


def content_digest(root: Path) -> tuple[str, int]:
    """sha256 over every file's relative path and bytes, in sorted order."""
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        h.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(path.read_bytes() + b"\0")
    return h.hexdigest(), len(files)


def verify() -> None:
    digest, n_files = content_digest(config.CORPUS_DIR)
    if digest != EXPECTED_DIGEST:
        raise SystemExit(
            f"content digest mismatch for {config.CORPUS_DIR}\n"
            f"  expected {EXPECTED_DIGEST}\n"
            f"  got      {digest}\n"
            "Delete that directory and the archive, then re-run."
        )
    n_pages = sum(1 for _ in config.DOCS_DIR.rglob("*.md"))
    n_src = sum(1 for p in config.DOCS_SRC_DIR.rglob("*") if p.is_file())
    print(f"\nready: {config.CORPUS_DIR}")
    print(f"  FastAPI {config.FASTAPI_VERSION} @ {config.FASTAPI_COMMIT[:10]}")
    print(f"  English pages      : {n_pages}")
    print(f"  docs_src files     : {n_src}")
    print(f"  content digest ok  : {n_files} files")
    print("\nnext:  python scripts/01_corpus_stats.py")


if __name__ == "__main__":
    download()
    if not config.DOCS_DIR.exists():
        print(f"extracting -> {config.CORPUS_DIR}")
        print(f"  {extract()} files")
    else:
        print(f"already extracted: {config.CORPUS_DIR}")
    verify()
