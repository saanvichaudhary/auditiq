"""ablation.py - retrieval ablation: chunking x embeddings x retriever x rerank. No LLM calls, so it's free and fast.

    python ablation.py --pdf apple_10k_2025.pdf --eval eval_set.json
    python ablation.py --pdf apple_10k_2025.pdf --eval eval_set.json --no-rerank --chunkers fixed_800 section_800

Metrics (per question, evidence = gold_phrases if given, else gold_pages):
  hit@k   : a relevant chunk is in the top k
  MRR     : mean reciprocal rank of the first relevant chunk (top 5)
  ctx_chars: average characters of the top-5 context (bigger chunks cost more tokens)
"""
import argparse
import csv
import json
import os
import re
import time

from chunking import CHUNKERS, describe_sections, make_chunks
from rag_engine import DEFAULT_CONFIG, extract_text_from_pdf, get_embeddings
from retrieval import HybridIndex

KS = (1, 3, 5)


def norm(s):
    return re.sub(r"\s+", "", s.lower())


def is_hit(doc, q):
    phrases = q.get("gold_phrases") or []
    if phrases:
        text = norm(doc.page_content)
        return any(norm(p) in text for p in phrases)
    return doc.metadata.get("page") in (q.get("gold_pages") or [])


def validate(pages, questions):
    """Drop questions that can never be scored, and say why. Prevents silently broken evals."""
    flat = [norm(p["text"]) for p in pages]
    n = max(p["page"] for p in pages)
    ok = []
    for q in questions:
        if not (q.get("gold_phrases") or q.get("gold_pages")):
            print(f"[skip] {q['id']}: fill in gold_phrases or gold_pages first")
            continue
        missing = [p for p in q.get("gold_phrases", []) if not any(norm(p) in t for t in flat)]
        if missing:
            print(f"[skip] {q['id']}: phrase(s) not found anywhere in the PDF text: {missing}")
            continue
        bad_pages = [p for p in q.get("gold_pages", []) if not 1 <= p <= n]
        if bad_pages:
            print(f"[skip] {q['id']}: gold_pages out of range: {bad_pages}")
            continue
        ok.append(q)
    return ok


def evaluate_config(index, questions, mode, rerank):
    stats = {f"hit@{k}": 0 for k in KS}
    rr, lat, ctx = 0.0, 0.0, 0.0
    for q in questions:
        t0 = time.perf_counter()
        docs = index.search(q["question"], k=max(KS), mode=mode, rerank=rerank)
        lat += time.perf_counter() - t0
        ctx += sum(len(d.page_content) for d in docs)
        first = next((i for i, d in enumerate(docs, 1) if is_hit(d, q)), None)
        if first:
            rr += 1.0 / first
            for k in KS:
                stats[f"hit@{k}"] += first <= k
    n = len(questions)
    row = {m: round(v / n, 3) for m, v in stats.items()}
    row.update({"MRR": round(rr / n, 3), "ctx_chars": int(ctx / n), "latency_ms": round(1000 * lat / n, 1)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--eval", default="eval_set.json")
    ap.add_argument("--embeddings", nargs="+",
                    default=[DEFAULT_CONFIG["embedding_model"], "BAAI/bge-small-en-v1.5"])
    ap.add_argument("--chunkers", nargs="+", default=list(CHUNKERS))
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pages = extract_text_from_pdf(args.pdf)
    if not pages:
        raise SystemExit("No text extracted - is this a scanned PDF?")
    print(f"Extracted {len(pages)} text pages.")
    with open(args.eval) as f:
        questions = validate(pages, json.load(f))
    if len(questions) < 8:
        print(f"WARNING: only {len(questions)} usable questions - results will be noisy. Aim for 25+.")
    if not questions:
        raise SystemExit("No usable questions.")
    print(f"Evaluating on {len(questions)} questions.\n")

    variants = [(m, False) for m in ("dense", "bm25", "hybrid")]
    if not args.no_rerank:
        variants += [("dense", True), ("hybrid", True)]

    rows = []
    for emb_name in args.embeddings:
        emb = get_embeddings(emb_name)
        for ch in args.chunkers:
            docs = make_chunks(pages, ch)
            if ch.startswith("section"):
                top = sorted(describe_sections(docs).items(), key=lambda x: -x[1])[:3]
                print(f"  [{ch}] sections detected: {len(describe_sections(docs))}, largest: {top}")
            index = HybridIndex(docs, emb)
            for mode, rerank in variants:
                if mode == "bm25" and emb_name != args.embeddings[0]:
                    continue  # BM25 ignores embeddings; run once
                r = evaluate_config(index, questions, mode, rerank)
                row = {"embedding": emb_name.split("/")[-1], "chunker": ch, "n_chunks": len(docs),
                       "retriever": mode + ("+rerank" if rerank else ""), **r}
                rows.append(row)
                print(f"{row['embedding']:<20} {ch:<18} {row['retriever']:<14} "
                      f"hit@1={r['hit@1']:.2f} hit@5={r['hit@5']:.2f} MRR={r['MRR']:.2f}")

    rows.sort(key=lambda r: (-r["MRR"], -r["hit@5"]))
    os.makedirs(args.out, exist_ok=True)
    cols = list(rows[0].keys())
    with open(os.path.join(args.out, "ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "ablation.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"\nSaved {args.out}/ablation.csv and ablation.md (sorted best-first).")


if __name__ == "__main__":
    main()
