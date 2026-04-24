# Artikate ML Engineer Assessment (Khushi Dua)

This repo contains my submission for the **Artikate AI/ML/LLM Engineer Technical Assessment**.

It includes:
- **Section 1**: Diagnosis of a failing LLM pipeline (written)
- **Section 2**: Production-grade **RAG** pipeline for legal PDFs + evaluation harness (precision@3)
- **Section 3**: Support ticket classifier (CPU, <500ms) + metrics + latency test
- **Section 4**: Written systems design answers (2/3 questions)

> Notes:
> - This project is designed to run locally in under ~5 minutes after installing dependencies.
> - Section 2 uses local embeddings + FAISS retrieval; generation uses **OpenAI** (`OPENAI_API_KEY` required).

---

## Quickstart

### Prerequisites
- Python 3.10+
- `pip`
- An OpenAI API key

### Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=...
```

### Run Section 2 demo (RAG)

```bash
python scripts/run_section2_demo.py
```

### Run Section 2 evaluation (precision@3)

```bash
python -m src.section2.evaluation
```

### Run Section 3 (train + eval + latency test)

```bash
python -m src.section3.run
```

### Run tests

```bash
pytest -q
```

---

## Repo Structure

```
.
├── ANSWERS.md
├── DESIGN.md
├── README.md
├── requirements.txt
├── .env.example
├── scripts/
│   └── run_section2_demo.py
├── src/
│   ├── section1/
│   ├── section2/
│   │   ├── rag_pipeline.py
│   │   └── evaluation.py
│   ├── section3/
│   │   ├── ticket_classifier.py
│   │   └── run.py
│   └── section4/
│       └── answers.md
└── tests/
    ├── test_section2_rag_eval.py
    └── test_section3_latency.py
```

---

## OpenAI configuration

Create `.env` from `.env.example`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

If you prefer a different model, set `OPENAI_MODEL`.

---

## Section 2 PDFs

This submission expects **3 sample PDFs** under:

`data/sample_pdfs/`

Place any three English PDF contracts/policies you like and rename them as:
- `nda.pdf`
- `service_agreement.pdf`
- `ip_agreement.pdf`

If you don’t have PDFs handy, you can quickly export 3 short sample PDFs from Google Docs / Word using any placeholder legal text.

The RAG pipeline will ingest them locally.

---
