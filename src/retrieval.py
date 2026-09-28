"""Four retrievers over one set of chunks, behind one interface.

    bm25            keyword matching (rank_bm25)
    dense           cosine similarity of bge-small embeddings, in numpy
    hybrid          both, merged by Reciprocal Rank Fusion
    hybrid+rerank   the hybrid top 20, reordered by a cross-encoder

Every retriever returns chunk indices, best first. Nothing here knows about
questions or evidence -- scoring lives in scoring.py -- so a retriever cannot
accidentally be tuned against the answers.
"""

from __future__ import annotations

import functools
import hashlib
import re
from collections.abc import Sequence

import numpy as np

from . import config
from .chunking import Chunk

RRF_K = 60  # the constant from the original RRF paper; not tuned here
FUSION_DEPTH = 50  # how deep into each list fusion looks
RERANK_DEPTH = 20  # how many hybrid results the cross-encoder reorders

# bge v1.5 is trained with this prefix on the *query* side of retrieval pairs,
# and its authors recommend it for short queries. Passages get no prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9_]+")


def bm25_tokens(text: str) -> list[str]:
    """Lower-cased words, keeping underscores.

    `response_model`, `status_code`, `HTTPException` are what users type and
    what the docs say; splitting them on the underscore would turn an exact
    identifier match into two weak common-word matches.
    """
    return _TOKEN.findall(text.lower())


class BM25:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.index = BM25Okapi([bm25_tokens(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[int]:
        scores = self.index.get_scores(bm25_tokens(query))
        return list(np.argsort(-scores, kind="stable")[:k])


@functools.lru_cache(maxsize=2)
def _sentence_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


class Dense:
    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], model: str = config.EMBED_MODEL) -> None:
        self.model_name = model
        self.matrix = self._embed_chunks([c.text for c in chunks])

    def _embed_chunks(self, texts: list[str]) -> np.ndarray:
        """Chunk embeddings, cached on disk by model and exact chunk texts.

        Encoding a chunking takes a minute on a CPU; the cache key covers every
        chunk's text, so a change to the chunker or the corpus re-encodes
        instead of silently reusing stale vectors.
        """
        digest = hashlib.sha256(
            (self.model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:20]
        path = config.INDEX_DIR / f"{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = _sentence_model(self.model_name).encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors.astype(np.float32))
        return np.load(path)

    def search(self, query: str, k: int) -> list[int]:
        q = _sentence_model(self.model_name).encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False)[0]
        # Normalised vectors, so the dot product is the cosine similarity.
        scores = self.matrix @ q
        return list(np.argsort(-scores, kind="stable")[:k])


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank).

    Uses only ranks, never raw scores, which is the point: BM25 scores are
    unbounded and cosine similarities sit between -1 and 1, so adding them would
    let whichever has the larger scale decide. Ties break towards the first
    list's order, so the result is deterministic.
    """
    score: dict[int, float] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for li, ranking in enumerate(rankings):
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(doc, (rank, li))
    return sorted(score, key=lambda d: (-score[d], first_seen[d]))


class Hybrid:
    name = "hybrid"

    def __init__(self, bm25: BM25, dense: Dense) -> None:
        self.bm25, self.dense = bm25, dense

    def search(self, query: str, k: int) -> list[int]:
        fused = rrf([self.bm25.search(query, FUSION_DEPTH), self.dense.search(query, FUSION_DEPTH)])
        return fused[:k]


@functools.lru_cache(maxsize=1)
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device="cpu")


class Reranked:
    """A cross-encoder reorders the top of another retriever's list.

    It reads query and chunk together, so it judges relevance far better than
    comparing two separately-computed vectors -- and costs a model call per
    candidate, which is why it only ever sees the top 20, never the corpus.
    """

    name = "hybrid+rerank"

    def __init__(self, first_stage: Hybrid, chunks: Sequence[Chunk],
                 model: str = config.RERANK_MODEL, depth: int = RERANK_DEPTH) -> None:
        self.first_stage, self.chunks, self.model_name, self.depth = first_stage, chunks, model, depth

    def search(self, query: str, k: int) -> list[int]:
        candidates = self.first_stage.search(query, self.depth)
        scores = _cross_encoder(self.model_name).predict(
            [(query, self.chunks[i].text) for i in candidates], show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), j))
        return [candidates[j] for j in order][:k]


def build(chunks: Sequence[Chunk]) -> dict[str, object]:
    """All four retrievers over one chunk list, sharing the expensive parts."""
    bm25, dense = BM25(chunks), Dense(chunks)
    hybrid = Hybrid(bm25, dense)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid,
            "hybrid+rerank": Reranked(hybrid, chunks)}
