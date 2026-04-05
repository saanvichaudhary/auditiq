"""
evaluator.py — RAG evaluation using Ragas metrics.
Run this separately after you've tested your QA chain to get a quality score.
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def run_evaluation(qa_chain, test_questions: list[str], ground_truths: list[str]) -> dict:
    """
    Evaluate RAG pipeline on a set of questions with known ground truths.
    
    Args:
        qa_chain: Built QA chain from rag_engine.py
        test_questions: List of questions to evaluate
        ground_truths: Expected answers for each question
    
    Returns:
        Dict with faithfulness, answer_relevancy, context_recall scores
    """
    answers = []
    contexts = []

    print("Running evaluation queries...")
    for q in test_questions:
        result = qa_chain.invoke({"query": q})
        answers.append(result["result"])
        contexts.append([doc.page_content for doc in result["source_documents"]])

    eval_dataset = Dataset.from_dict({
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Computing Ragas scores (faithfulness, relevancy, context recall)...")
    scores = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    return scores


# --- Sample test set for a 10-K report ---
SAMPLE_QUESTIONS = [
    "What are the primary audit risks identified in this report?",
    "How does the company recognize revenue?",
    "What going concern disclosures are made?",
    "What are the key internal control weaknesses noted?",
    "What is the company's total revenue for the reported period?",
]

SAMPLE_GROUND_TRUTHS = [
    "Audit risks vary by company — check the Critical Audit Matters section.",
    "Revenue recognition follows ASC 606 or IFRS 15 depending on jurisdiction.",
    "Going concern disclosures appear in auditor notes if there is doubt.",
    "Internal control weaknesses are disclosed under SOX Section 302/404.",
    "Total revenue is reported in the consolidated income statement.",
]


if __name__ == "__main__":
    # Example usage — import your chain and run
    from rag_engine import extract_text_from_pdf, build_vectorstore, build_qa_chain

    PDF_PATH = "sample_report.pdf"  # Replace with your test PDF
    pages = extract_text_from_pdf(PDF_PATH)
    vectorstore, _ = build_vectorstore(pages)
    qa_chain = build_qa_chain(vectorstore)

    scores = run_evaluation(qa_chain, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTHS)
    print("\n=== AuditIQ RAG Evaluation Results ===")
    print(f"Faithfulness:      {scores['faithfulness']:.2%}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.2%}")
    print(f"Context Recall:    {scores['context_recall']:.2%}")