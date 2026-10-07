   import json
   from ablation import is_hit, validate
   from chunking import make_chunks
   from rag_engine import extract_text_from_pdf, get_embeddings
   from retrieval import HybridIndex

   pages = extract_text_from_pdf("apple_10-k_source_pdf.pdf")
   qs = validate(pages, json.load(open("eval_set.json")))
   idx = HybridIndex(make_chunks(pages, "section_800"),
                     get_embeddings("sentence-transformers/all-MiniLM-L6-v2"))
   for mode in ("dense", "bm25", "hybrid"):
       print(f"\n=== {mode} ===")
       for q in qs:
           docs = idx.search(q["question"], k=5, mode=mode)
           rank = next((i for i, d in enumerate(docs, 1) if is_hit(d, q)), None)
           print(f"{q['id']}  first_hit_rank={rank}")