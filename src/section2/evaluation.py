from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from src.section2.rag_pipeline import RAGPipeline


@dataclass
class EvalItem:
    question: str
    expected_doc_contains: str
    expected_keywords: List[str]


EVAL_SET: List[EvalItem] = [
    EvalItem("What is the notice period?", "nda", ["notice", "days", "written"]),
    EvalItem("What is the governing law?", "service", ["govern", "law", "jurisdiction"]),
    EvalItem("What are the payment terms?", "service", ["payment", "invoice", "due"]),
    EvalItem("What is the limitation of liability?", "nda", ["liability", "damages", "limit"]),
    EvalItem("What happens upon termination?", "service", ["termination", "terminate", "survive"]),
    EvalItem("What confidentiality obligations are described?", "nda", ["confidential", "disclosure", "protect"]),
    EvalItem("Is there an indemnification clause?", "service", ["indemn", "hold harmless", "claim"]),
    EvalItem("Are there insurance requirements?", "service", ["insurance", "coverage", "liability"]),
    EvalItem("What IP/license rights are granted?", "ip", ["license", "intellectual", "rights"]),
    EvalItem("Does the agreement mention renewal?", "service", ["renew", "term", "period"]),
]


def precision_at_k(pipeline: RAGPipeline, k: int = 3) -> Dict[str, Any]:
    hits = 0
    per_q = []

    for item in EVAL_SET:
        res = pipeline.query(item.question)
        top_sources = res.sources[:k]

        found = False
        for s in top_sources:
            doc_ok = item.expected_doc_contains.lower() in s["document"].lower()
            kw_ok = any(kw.lower() in s["chunk"].lower() for kw in item.expected_keywords)
            if doc_ok and kw_ok:
                found = True
                break

        hits += 1 if found else 0
        per_q.append(
            {
                "question": item.question,
                "hit": found,
                "top_docs": [(s["document"], s["page"]) for s in top_sources],
            }
        )

    return {
        "k": k,
        "total": len(EVAL_SET),
        "hits": hits,
        "precision_at_k": hits / len(EVAL_SET),
        "per_question": per_q,
    }


def main() -> None:
    pipeline = RAGPipeline()
    report = precision_at_k(pipeline, k=3)

    print(f"precision@{report['k']}: {report['precision_at_k']:.2%} ({report['hits']}/{report['total']})")
    for i, row in enumerate(report["per_question"], start=1):
        status = "PASS" if row["hit"] else "FAIL"
        print(f"{i:02d}. {status} | {row['question']} | top={row['top_docs']}")


if __name__ == "__main__":
    main()
