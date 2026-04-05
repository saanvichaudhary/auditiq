import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def build_vectorstore(pages: list[dict]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    docs = []
    for p in pages:
        chunks = splitter.create_documents(
            texts=[p["text"]],
            metadatas=[{"page": p["page"]}]
        )
        docs.extend(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, docs


def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

    prompt_template = """You are AuditIQ, an AI assistant that helps analysts 
understand financial and audit documents. Answer the question using ONLY the 
context provided below. If the context does not contain enough information to 
answer confidently, say: "I'm not confident — the document may not cover this. 
Please verify manually."

Always cite the page number(s) your answer draws from.

Context:
{context}

Question: {question}

Answer (with page citations):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def ask(qa_chain, question: str) -> dict:
    chain = qa_chain["chain"]
    retriever = qa_chain["retriever"]

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    pages_cited = sorted(set(doc.metadata["page"] for doc in source_docs))
    low_confidence = "not confident" in answer.lower() or "verify manually" in answer.lower()

    return {
        "answer": answer,
        "pages_cited": pages_cited,
        "low_confidence": low_confidence,
        "source_chunks": [doc.page_content[:300] for doc in source_docs]
    }