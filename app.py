import streamlit as st
import tempfile
import os
from rag_engine import extract_text_from_pdf, build_vectorstore, build_qa_chain, ask

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AuditIQ",
    page_icon="🔍",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='color:#1F4E79; margin-bottom:0'>🔍 AuditIQ</h1>
    <p style='color:#555; margin-top:4px; font-size:16px'>
        AI-Assisted Financial Document Intelligence &nbsp;·&nbsp; 
        RAG-powered &nbsp;·&nbsp; Source-cited &nbsp;·&nbsp; Hallucination-aware
    </p>
    <hr style='border:1px solid #e0e0e0; margin-bottom:24px'>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload an annual report, 10-K, or audit document (PDF)",
        type=["pdf"]
    )
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    example_qs = [
        "What are the key audit risks identified?",
        "How does the company recognize revenue?",
        "What going concern disclosures are made?",
        "Summarise the internal control weaknesses.",
        "What is the total revenue for the period?",
        "What changes were made to accounting policies?",
    ]
    for q in example_qs:
        if st.button(q, use_container_width=True):
            st.session_state["question_input"] = q

    st.markdown("---")
    st.markdown(
        "<small style='color:#aaa'>Built by Saanvi Chaudhary · "
        "LangChain · Gemini Flash · FAISS · Ragas</small>",
        unsafe_allow_html=True
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

# ── PDF Processing ────────────────────────────────────────────────────────────
if uploaded_file:
    if st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("📖 Reading and indexing document..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            pages = extract_text_from_pdf(tmp_path)
            os.unlink(tmp_path)

            if not pages:
                st.error("Could not extract text from this PDF. It may be scanned/image-based.")
            else:
                vectorstore, docs = build_vectorstore(pages)
                st.session_state["qa_chain"] = build_qa_chain(vectorstore)
                st.session_state["last_file"] = uploaded_file.name
                st.session_state["history"] = []
                st.session_state["page_count"] = len(pages)
                st.success(f"✅ Indexed {len(pages)} pages · {len(docs)} chunks · Ready to query")

# ── Q&A Interface ─────────────────────────────────────────────────────────────
if st.session_state["qa_chain"]:
    st.markdown("### Ask a question about the document")

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Question",
            value=st.session_state["question_input"],
            placeholder="e.g. What are the primary audit risks flagged this year?",
            label_visibility="collapsed",
            key="q_input"
        )
    with col2:
        submit = st.button("Ask →", use_container_width=True, type="primary")

    if submit and question.strip():
        st.session_state["question_input"] = ""
        with st.spinner("Thinking..."):
            result = ask(st.session_state["qa_chain"], question)

        st.session_state["history"].insert(0, {"question": question, "result": result})

    # ── Render history ────────────────────────────────────────────────────────
    for item in st.session_state["history"]:
        q = item["question"]
        r = item["result"]

        st.markdown(f"**Q: {q}**")

        # Confidence badge
        if r["low_confidence"]:
            st.warning("⚠️ Low confidence — answer may be incomplete. Verify manually.")
        else:
            st.success("✅ High confidence")

        st.markdown(r["answer"])

        # Source pages
        pages_str = ", ".join(f"p.{p}" for p in r["pages_cited"])
        st.markdown(f"<small style='color:#888'>📄 Sources: {pages_str}</small>", unsafe_allow_html=True)

        # Expandable chunks
        with st.expander("View retrieved context chunks"):
            for i, chunk in enumerate(r["source_chunks"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.text(chunk)

        st.markdown("---")

else:
    # Placeholder when no file uploaded
    st.markdown("""
    <div style='text-align:center; padding:60px; color:#aaa'>
        <h3>👈 Upload a PDF to get started</h3>
        <p>Supports annual reports, 10-K filings, audit documents, prospectuses</p>
    </div>
    """, unsafe_allow_html=True)