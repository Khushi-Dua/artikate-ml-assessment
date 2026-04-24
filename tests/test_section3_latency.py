from src.section3.ticket_classifier import DistilBertTicketClassifier, synthetic_data, split_data

def test_classifier_latency_under_500ms():
    examples = synthetic_data(20)  # tiny quick train
    train, _ = split_data(examples, test_size=0.5)

    clf = DistilBertTicketClassifier()
    clf.fit(train, epochs=1, batch_size=8)

    pred, conf, ms = clf.predict_one("I was charged twice this month.")
    assert pred in {"billing", "technical_issue", "feature_request", "complaint", "other"}
    assert ms < 500.0
