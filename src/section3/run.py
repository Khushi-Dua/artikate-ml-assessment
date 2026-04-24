from __future__ import annotations

import numpy as np

from src.section3.ticket_classifier import (
    DistilBertTicketClassifier,
    synthetic_data,
    split_data,
    CLASSES,
)


def latency_test(clf: DistilBertTicketClassifier, n: int = 20, budget_ms: float = 500.0):
    samples = [
        "I was charged twice this month.",
        "The export button does nothing.",
        "Can you add a dark mode?",
        "Your service is terrible and I'm unhappy.",
        "What are your support hours?",
    ]
    times = []
    for i in range(n):
        pred, conf, ms = clf.predict_one(samples[i % len(samples)])
        assert pred in CLASSES
        times.append(ms)
        assert ms < budget_ms, f"Latency {ms:.1f}ms exceeds {budget_ms}ms"

    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": float(np.max(times)),
        "budget_ms": budget_ms,
    }


def main() -> None:
    examples = synthetic_data(200)
    train, test = split_data(examples)

    clf = DistilBertTicketClassifier()
    clf.fit(train, epochs=2, batch_size=16)

    metrics = clf.evaluate(test)
    print("Accuracy:", metrics.accuracy)
    print("Macro F1:", metrics.macro_f1)
    print("Per-class F1:", metrics.per_class_f1)
    print("Confusion matrix:\n", metrics.confusion)
    print("\nClassification report:\n", metrics.report)

    lat = latency_test(clf)
    print("\nLatency:", lat)


if __name__ == "__main__":
    main()
