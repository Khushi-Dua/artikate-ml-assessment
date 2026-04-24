from __future__ import annotations

from src.section2.rag_pipeline import RAGPipeline

def main():
    pipeline = RAGPipeline(pdf_dir="data/sample_pdfs")

    questions = [
        "What is the notice period?",
        "What is the limitation of liability?",
        "What are the payment terms?",
        "What is the governing law?",
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print("Q:", q)
        res = pipeline.query(q)
        print("Confidence:", round(res.confidence, 3))
        print("Answer:\n", res.answer)
        print("\nTop sources:")
        for s in res.sources:
            print(f"- {s['document']} (page {s['page']})")

if __name__ == "__main__":
    main()
