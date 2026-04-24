from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW


CLASSES = ["billing", "technical_issue", "feature_request", "complaint", "other"]
C2I = {c: i for i, c in enumerate(CLASSES)}
I2C = {i: c for c, i in C2I.items()}


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    confusion: np.ndarray
    report: str


class DistilBertTicketClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        self.device = torch.device("cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(CLASSES)
        ).to(self.device)

    def _encode(self, texts: List[str]):
        return self.tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )

    def fit(self, examples: List[Dict[str, str]], epochs: int = 3, batch_size: int = 16, lr: float = 2e-5) -> None:
        X = [e["text"] for e in examples]
        y = [C2I[e["label"]] for e in examples]

        enc = self._encode(X)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        for _ in range(epochs):
            for input_ids, attn, labels in dl:
                input_ids = input_ids.to(self.device)
                attn = attn.to(self.device)
                labels = labels.to(self.device)

                opt.zero_grad()
                out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
                out.loss.backward()
                opt.step()

    def predict_one(self, text: str) -> Tuple[str, float, float]:
        start = time.time()
        self.model.eval()

        enc = self._encode([text])
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        ms = (time.time() - start) * 1000.0
        return I2C[idx], conf, ms

    def evaluate(self, examples: List[Dict[str, str]]) -> Metrics:
        y_true = np.array([C2I[e["label"]] for e in examples], dtype=int)
        y_pred = []
        for e in examples:
            pred, _, _ = self.predict_one(e["text"])
            y_pred.append(C2I[pred])
        y_pred = np.array(y_pred, dtype=int)

        acc = float(accuracy_score(y_true, y_pred))
        per = f1_score(y_true, y_pred, average=None, zero_division=0)
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        rep = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)

        return Metrics(
            accuracy=acc,
            macro_f1=macro,
            per_class_f1={CLASSES[i]: float(per[i]) for i in range(len(CLASSES))},
            confusion=cm,
            report=rep,
        )


def synthetic_data(n_per_class: int = 200) -> List[Dict[str, str]]:
    templates = {
        "billing": [
            "I was charged twice for my subscription.",
            "My refund hasn't arrived yet.",
            "Please send me an invoice for last month.",
            "I need to update my payment method.",
        ],
        "technical_issue": [
            "The app crashes when I click export.",
            "The login page shows an error.",
            "The dashboard is not loading.",
            "Upload fails with a timeout.",
        ],
        "feature_request": [
            "Please add dark mode.",
            "Can you add SSO support?",
            "I would like an API for exporting data.",
            "Add a bulk edit option.",
        ],
        "complaint": [
            "This is unacceptable and very frustrating.",
            "Your support team is not responding.",
            "Worst experience I have had.",
            "The product quality is terrible.",
        ],
        "other": [
            "Where can I find documentation?",
            "What are your support hours?",
            "Do you offer training?",
            "How do I contact sales?",
        ],
    }

    out = []
    for label, ts in templates.items():
        for i in range(n_per_class):
            out.append({"text": ts[i % len(ts)], "label": label})
    return out


def split_data(examples: List[Dict[str, str]], test_size: float = 0.2):
    X = [e["text"] for e in examples]
    y = [e["label"] for e in examples]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    train = [{"text": t, "label": l} for t, l in zip(X_tr, y_tr)]
    test = [{"text": t, "label": l} for t, l in zip(X_te, y_te)]
    return train, test
