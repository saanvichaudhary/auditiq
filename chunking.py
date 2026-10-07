"""chunking.py - chunking strategies for 10-K style PDFs.

Every chunk is a langchain Document with metadata: page (1-based PDF page), section.
Chunks never cross page boundaries, so page citations stay exact.
"""
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Matches headings such as "Item 1A. Risk Factors" at the start of a line.
ITEM_RE = re.compile(r"^\s*Item\s+(\d+[A-C]?)\.\s*(.*)$", re.IGNORECASE)

CHUNKERS = {
    "page":              {"kind": "page"},
    "fixed_500":         {"kind": "fixed",   "size": 500,  "overlap": 75},
    "fixed_800":         {"kind": "fixed",   "size": 800,  "overlap": 150},
    "fixed_1200":        {"kind": "fixed",   "size": 1200, "overlap": 200},
    "section_800_nohdr": {"kind": "section", "size": 800,  "overlap": 150, "header": False},
    "section_800":       {"kind": "section", "size": 800,  "overlap": 150, "header": True},
    "section_1200":      {"kind": "section", "size": 1200, "overlap": 200, "header": True},
}


def _splitter(size, overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def section_segments(pages):
    """Yield (page_number, [(section_label, text), ...]) tracking the running Item heading."""
    current = "Front matter"
    for p in pages:
        segs, buf, seg_section = [], [], current
        for line in p["text"].splitlines():
            m = ITEM_RE.match(line)
            if m and len(line.strip()) < 90:
                if buf:
                    segs.append((seg_section, "\n".join(buf)))
                buf = []
                current = f"Item {m.group(1).upper()}. {m.group(2).strip()}".strip()
                seg_section = current
            buf.append(line)
        if buf:
            segs.append((seg_section, "\n".join(buf)))
        yield p["page"], segs


def make_chunks(pages, strategy):
    """pages: list of {"page": int, "text": str}. Returns list[Document]."""
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunker '{strategy}'. Options: {list(CHUNKERS)}")
    cfg = CHUNKERS[strategy]
    docs = []

    if cfg["kind"] == "page":
        for p in pages:
            docs.append(Document(page_content=p["text"].strip(),
                                 metadata={"page": p["page"], "section": ""}))
    elif cfg["kind"] == "fixed":
        sp = _splitter(cfg["size"], cfg["overlap"])
        for p in pages:
            for t in sp.split_text(p["text"]):
                if t.strip():
                    docs.append(Document(page_content=t.strip(),
                                         metadata={"page": p["page"], "section": ""}))
    else:  # section-aware
        sp = _splitter(cfg["size"], cfg["overlap"])
        for page, segs in section_segments(pages):
            for section, text in segs:
                for t in sp.split_text(text):
                    if not t.strip():
                        continue
                    body = t.strip()
                    if cfg.get("header"):
                        body = f"[{section} | p.{page}]\n{body}"
                    docs.append(Document(page_content=body,
                                         metadata={"page": page, "section": section}))
    return docs


def describe_sections(docs):
    """Chunk counts per section - a quick sanity check that heading detection works."""
    counts = {}
    for d in docs:
        s = d.metadata.get("section", "")
        counts[s] = counts.get(s, 0) + 1
    return counts
