"""rag_engine.py - AuditIQ v2 core: extraction, indexing, retrieval, grounded generation.

Public API is unchanged from v1, so app.py keeps working as-is:
    extract_text_from_pdf(path) -> pages
    build_vectorstore(pages)    -> (index, docs)
    build_qa_chain(index)       -> qa dict
    ask(qa, question)           -> dict(answer, pages_cited, low_confidence, source_chunks, ...)
"""
import os
import re
import time
from functools import lru_cache

import fitz  # PyMuPDF
from dotenv import load_dotenv

from chunking import make_chunks
from retrieval import HybridIndex

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Update these after you read your ablation results.
DEFAULT_CONFIG = {
    "chunker": "section_800",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "mode": "hybrid",
    "rerank": True,
    "k": 5,
    "abstain_below": 0.35,   # best-chunk cosine below this -> don't even call the LLM (tune on your data)
    "low_conf_below": 0.50,  # below this -> show the low-confidence badge
}

REFUSAL = "I'm not confident — the document may not cover this. Please verify manually."

PROMPT = """You are AuditIQ, an assistant that helps analysts understand financial and audit documents.
Answer the question using ONLY the numbered excerpts below.
Cite the excerpt number in square brackets right after each claim, e.g. [1] or [2][4].
Quote figures exactly as written in the excerpts, with units and period.
If the excerpts do not contain enough information, reply exactly: "{refusal}"

Excerpts:
{context}

Question: {question}

Answer:"""


@lru_cache(maxsize=4)
def get_embeddings(model_name):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},  # required for top_cosine()
    )


def extract_text_from_pdf(pdf_path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


def build_vectorstore(pages, config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    docs = make_chunks(pages, cfg["chunker"])
    index = HybridIndex(docs, get_embeddings(cfg["embedding_model"]))
    index.config = cfg
    return index, docs


def _default_llm():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in a .env file.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                                  google_api_key=GEMINI_API_KEY, temperature=0.1)


def build_qa_chain(index, llm=None, config=None):
    cfg = {**DEFAULT_CONFIG, **(index.config or {}), **(config or {})}
    return {"index": index, "llm": llm or _default_llm(), "config": cfg}


def _resp_text(resp):
    c = resp.content
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def ask(qa, question):
    index, llm, cfg = qa["index"], qa["llm"], qa["config"]
    docs = index.search(question, k=cfg["k"], mode=cfg["mode"], rerank=cfg["rerank"])
    top_cos = index.top_cosine(question)
    base = {"retrieved_docs": docs, "top_cosine": top_cos, "error": None,
            "source_chunks": [d.page_content[:300] for d in docs]}

    if top_cos < cfg["abstain_below"]:
        return {**base, "answer": REFUSAL, "pages_cited": [], "cited_chunk_ids": [],
                "low_confidence": True, "abstained": True}

    context = "\n\n".join(
        f"[{i}] (page {d.metadata['page']}) {d.page_content}" for i, d in enumerate(docs, 1)
    )
    prompt = PROMPT.format(refusal=REFUSAL, context=context, question=question)
    raw = None
    for attempt in range(3):
        try:
            raw = _resp_text(llm.invoke(prompt))
            break
        except Exception as e:  # surface API/key/quota problems instead of crashing the UI
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 2:
                time.sleep(5 * (attempt + 1))  # model overloaded: wait and retry
                continue
            return {**base, "answer": f"LLM error: {e}", "pages_cited": [], "cited_chunk_ids": [],
                    "low_confidence": True, "abstained": False, "error": str(e)}

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", raw) if 1 <= int(n) <= len(docs)})
    cited_docs = [docs[n - 1] for n in cited]
    # fall back to all retrieved pages only if the model cited nothing
    pages = sorted({d.metadata["page"] for d in (cited_docs or docs)})
    shown = re.sub(r"\[(\d+)\]",
                   lambda m: f"[p.{docs[int(m.group(1)) - 1].metadata['page']}]"
                   if 1 <= int(m.group(1)) <= len(docs) else m.group(0), raw)
    refused = "not confident" in raw.lower() or "verify manually" in raw.lower()
    if refused:
        pages, cited = [], []
    return {**base, "answer": shown, "pages_cited": pages, "cited_chunk_ids": cited,
            "low_confidence": refused or top_cos < cfg["low_conf_below"],
            "abstained": False}
