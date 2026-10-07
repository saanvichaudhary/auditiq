"""find_pages.py - help write the eval set: find which PDF pages contain a phrase.

    python find_pages.py apple_10k_2025.pdf "total net sales"
"""
import re
import sys
from rag_engine import extract_text_from_pdf

if len(sys.argv) < 3:
    sys.exit('usage: python find_pages.py <pdf> "<phrase>"')
pages = extract_text_from_pdf(sys.argv[1])
needle = re.sub(r"\s+", "", sys.argv[2].lower())
hits = 0
for p in pages:
    flat = re.sub(r"\s+", " ", p["text"])
    if needle in re.sub(r"\s+", "", flat.lower()):
        hits += 1
        i = flat.lower().find(sys.argv[2].lower())
        snippet = flat[max(0, i - 80): i + 160] if i >= 0 else flat[:200]
        print(f"--- PDF page {p['page']} ---\n{snippet}\n")
print(f"{hits} page(s) matched.")
