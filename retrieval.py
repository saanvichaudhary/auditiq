"""retrieval.py - dense (FAISS), sparse (BM25) and hybrid (RRF) retrieval + optional cross-encoder rerank."""
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text):
    # keeps numbers like 383,285 or 1.5 as single tokens, which matters for financial text
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_NAME)


class HybridIndex:
    MODES = ("dense", "bm25", "hybrid")

    def __init__(self, docs, embeddings):
        if not docs:
            raise ValueError("No chunks to index.")
        for i, d in enumerate(docs):
            d.metadata["cid"] = i
        self.docs = docs
        self.faiss = FAISS.from_documents(docs, embeddings)
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])
        self.config = {}

    def _dense_ids(self, q, n):
        res = self.faiss.similarity_search_with_score(q, n)
        return [int(d.metadata["cid"]) for d, _ in res]

    def _bm25_ids(self, q, n):
        scores = self.bm25.get_scores(tokenize(q))
        return [int(i) for i in np.argsort(scores)[::-1][:n]]

    def top_cosine(self, q):
        """Cosine similarity of the best dense match. Needs normalized embeddings:
        FAISS returns squared L2 distance d, and for unit vectors cos = 1 - d/2."""
        _, dist = self.faiss.similarity_search_with_score(q, 1)[0]
        return float(1.0 - dist / 2.0)

    def search(self, q, k=5, mode="hybrid", rerank=False, pool=20):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        n = max(pool, k) if rerank else k
        if mode == "dense":
            ids = self._dense_ids(q, n)
        elif mode == "bm25":
            ids = self._bm25_ids(q, n)
        else:  # reciprocal rank fusion
            fused = {}
            for ranked in (self._dense_ids(q, pool), self._bm25_ids(q, pool)):
                for r, cid in enumerate(ranked):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + r + 1)
            ids = sorted(fused, key=fused.get, reverse=True)[:n]
        if rerank and ids:
            scores = _reranker().predict([(q, self.docs[i].page_content) for i in ids])
            ids = [ids[j] for j in np.argsort(scores)[::-1]]
        return [self.docs[i] for i in ids[:k]]
