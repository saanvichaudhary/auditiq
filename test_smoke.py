"""Offline tests (no model downloads, no API key): run with `pytest -q` from the project root."""
import os
import sys

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chunking import CHUNKERS, make_chunks                      # noqa: E402
from retrieval import HybridIndex, tokenize                      # noqa: E402
from rag_engine import REFUSAL, ask, build_qa_chain              # noqa: E402
from ablation import is_hit, validate                            # noqa: E402

PAGES = [
    {"page": 1, "text": "Apple Inc. Annual Report\nTable of contents\n"},
    {"page": 2, "text": "Item 1A. Risk Factors\n" + "Tariffs could harm margins. " * 60},
    {"page": 3, "text": "Item 7. Management's Discussion\nTotal net sales were $416,161 million in 2025.\n" + "Filler text. " * 80},
]


class HashEmb(Embeddings):
    def _v(self, t):
        v = np.zeros(64)
        for w in tokenize(t):
            v[sum(map(ord, w)) % 64] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, t):
        return self._v(t)


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def invoke(self, prompt):
        return type("R", (), {"content": self.text})()


@pytest.mark.parametrize("name", list(CHUNKERS))
def test_every_chunker_produces_valid_chunks(name):
    docs = make_chunks(PAGES, name)
    assert docs and all(d.page_content.strip() for d in docs)
    assert {d.metadata["page"] for d in docs} <= {1, 2, 3}


def test_section_labels_and_headers():
    docs = make_chunks(PAGES, "section_800")
    p2 = [d for d in docs if d.metadata["page"] == 2]
    assert all(d.metadata["section"].startswith("Item 1A") for d in p2)
    assert p2[0].page_content.startswith("[Item 1A")
    assert not any(d.page_content.startswith("[") for d in make_chunks(PAGES, "section_800_nohdr"))


@pytest.fixture(scope="module")
def index():
    return HybridIndex(make_chunks(PAGES, "fixed_500"), HashEmb())


@pytest.mark.parametrize("mode", HybridIndex.MODES)
def test_search_modes_return_k(index, mode):
    assert len(index.search("tariffs margins", k=3, mode=mode)) == 3


def test_bm25_finds_exact_number(index):
    top = index.search("416,161", k=1, mode="bm25")[0]
    assert "416,161" in top.page_content


def test_hit_and_validate():
    d = make_chunks(PAGES, "page")[2]
    assert is_hit(d, {"gold_phrases": ["$416,161 million"]})
    assert is_hit(d, {"gold_pages": [3]}) and not is_hit(d, {"gold_pages": [2]})
    qs = [{"id": "a", "question": "x", "gold_phrases": ["not in pdf"]},
          {"id": "b", "question": "x"},
          {"id": "c", "question": "x", "gold_phrases": ["416,161"]}]
    assert [q["id"] for q in validate(PAGES, qs)] == ["c"]


def test_ask_parses_citations_and_pages(index):
    qa = build_qa_chain(index, llm=FakeLLM("Net sales were $416,161 million [1]."),
                        config={"k": 3, "mode": "bm25", "abstain_below": -1})
    r = ask(qa, "total net sales 416,161")
    assert r["cited_chunk_ids"] == [1] and "[p." in r["answer"] and r["pages_cited"]
    assert r["error"] is None and set(r) >= {"answer", "pages_cited", "low_confidence", "source_chunks"}


def test_abstain_skips_llm(index):
    class Boom:
        def invoke(self, _):
            raise AssertionError("LLM should not be called")
    qa = build_qa_chain(index, llm=Boom(), config={"abstain_below": 2.0})
    r = ask(qa, "anything")
    assert r["answer"] == REFUSAL and r["abstained"] and r["low_confidence"]


def test_llm_error_is_surfaced(index):
    class Bad:
        def invoke(self, _):
            raise RuntimeError("quota")
    qa = build_qa_chain(index, llm=Bad(), config={"abstain_below": -1})
    r = ask(qa, "tariffs")
    assert r["error"] == "quota" and r["low_confidence"]
