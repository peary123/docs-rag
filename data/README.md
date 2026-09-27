# data/

This directory is gitignored. It is regenerated from a pinned FastAPI commit by

    python scripts/00_prepare_data.py

which downloads the source archive of FastAPI 0.141.1 (commit `95f8322e`,
~18 MB) and extracts:

    data/fastapi-0.141.1/docs/en/docs/**/*.md   the English pages
    data/fastapi-0.141.1/docs_src/              code the pages include at build time
    data/fastapi-0.141.1/fastapi/               the library; one page includes from it
    data/fastapi-0.141.1/LICENSE

The extracted files are checked against a digest of their contents, not
against a checksum of the archive (see NOTES.md for why).

Source: https://github.com/fastapi/fastapi (MIT License)
