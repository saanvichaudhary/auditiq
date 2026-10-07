"""evaluator.py - end-to-end evaluation (retrieval + Gemini answer) using the real rag_engine pipeline.

    python evaluator.py --pdf apple_10k_2025.pdf --eval eval_set.json --chunker section_800 --mode hybrid
    python evaluator.py ... --ragas        # optional LLM-judged metrics (needs a reference "answer" per question)

Per-question checks:
  answer_match : any answer_phrases (or gold_phrases for numeric questions) appears in the answer
  citation_ok  : a cited chunk is relevant (same rule as ablation.py)
  abstained    : system refused / low confidence
"""
import argparse
import json
import time

from ablation import is_hit, norm, validate
from rag_engine import DEFAULT_CONFIG, ask, build_qa_chain, build_vectorstore, extract_text_from_pdf


def run(qa, questions, sleep=2.0):
    rows = []
    for q in questions:
        r = ask(qa, q["question"])
        if r["error"]:
            print(f"[error] {q['id']}: {r['error']}")
        cited_docs = [r["retrieved_docs"][n - 1] for n in r["cited_chunk_ids"]]
        phrases = q.get("answer_phrases") or (q.get("gold_phrases") if q.get("type") == "numeric" else [])
        rows.append({
            "id": q["id"], "question": q["question"], "answer": r["answer"],
            "answer_match": (any(norm(p) in norm(r["answer"]) for p in phrases) if phrases else None),
            "citation_ok": any(is_hit(d, q) for d in cited_docs),
            "retrieval_hit": any(is_hit(d, q) for d in r["retrieved_docs"]),
            "abstained": r["low_confidence"], "top_cosine": round(r["top_cosine"], 3),
            "contexts": [d.page_content for d in r["retrieved_docs"]], "error": r["error"],
        })
        time.sleep(sleep)  # be gentle with free-tier rate limits
    return rows


def summarize(rows):
    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{sum(vals)}/{len(vals)} = {sum(vals) / len(vals):.0%}" if vals else "n/a"
    print("\n=== End-to-end results ===")
    for key in ("retrieval_hit", "citation_ok", "answer_match", "abstained"):
        print(f"{key:<14} {rate(key)}")
    print(f"errors         {sum(1 for r in rows if r['error'])}")


def run_ragas(rows, questions):
    """Optional. Ragas' API changes between versions; this targets the 0.2+ style - adjust if yours differs."""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from langchain_google_genai import ChatGoogleGenerativeAI
        from rag_engine import GEMINI_API_KEY, get_embeddings
        ref = {q["id"]: q.get("answer", "") for q in questions}
        ds = EvaluationDataset.from_list([
            {"user_input": r["question"], "response": r["answer"],
             "retrieved_contexts": r["contexts"], "reference": ref[r["id"]]}
            for r in rows if not r["abstained"]])
        llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY))
        emb = LangchainEmbeddingsWrapper(get_embeddings(DEFAULT_CONFIG["embedding_model"]))
        print(evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy()], llm=llm, embeddings=emb))
    except Exception as e:
        print(f"[ragas skipped] {type(e).__name__}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--eval", default="eval_set.json")
    ap.add_argument("--chunker", default=DEFAULT_CONFIG["chunker"])
    ap.add_argument("--mode", default=DEFAULT_CONFIG["mode"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--out", default="results/e2e.json")
    a = ap.parse_args()

    pages = extract_text_from_pdf(a.pdf)
    with open(a.eval) as f:
        qs = validate(pages, json.load(f))
    index, _ = build_vectorstore(pages, {"chunker": a.chunker})
    qa = build_qa_chain(index, config={"mode": a.mode, "rerank": a.rerank})
    rows = run(qa, qs, a.sleep)
    summarize(rows)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {a.out}")
    if a.ragas:
        run_ragas(rows, qs)
