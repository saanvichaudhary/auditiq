# AuditIQ — AI-Assisted Financial Document Intelligence

RAG-based QA system over financial reports and 10-K filings.

## Stack
- LangChain + Gemini 2.5 Flash — LLM backbone
- FAISS — vector similarity search
- HuggingFace Embeddings — local, free embeddings
- Ragas — RAG evaluation (faithfulness, relevancy, context recall)
- Streamlit — UI

## Features
- Upload any annual report or 10-K PDF
- Ask natural language questions with source-cited answers
- Confidence flagging for low-retrieval answers
- RAG evaluation pipeline

## Run locally
pip install -r requirements.txt
streamlit run app.py