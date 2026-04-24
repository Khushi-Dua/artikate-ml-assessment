# DESIGN.md — Section 2: Production‑Grade RAG Pipeline (Legal PDFs)

## Goals & Constraints
- Corpus: 500+ English PDF contracts/policies (~40 pages average)
- Queries: precise clause-level questions (notice period, liability caps, etc.)
- Output must cite **document + page**
- Hallucination unacceptable → must refuse when context insufficient

---

## Architecture Overview
1) PDF ingestion → page text extraction  
2) Chunking with overlap + metadata (doc, page)  
3) Embeddings (local model)  
4) Vector store (FAISS local)  
5) Retrieval (top-k)  
6) Generation (OpenAI) with citation constraints  
7) Hallucination mitigation (confidence + refusal)

---

## Chunking strategy
**Choice**: page-aware, sliding window chunking: ~512 tokens with ~64 token overlap.

**Why**
- Legal clauses often span multiple sentences; overlap prevents boundary loss.
- Page-level metadata is required for citations. Chunk boundaries should not lose “page number”.

**Tradeoffs**
- Fixed windows are simple and consistent but can split semantic sections; overlap mitigates.
- Better alternative at scale: section/heading-aware chunking + paragraph boundaries.

---

## Embedding model choice
**Choice**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast on CPU)

**Why**
- Works offline (no API), quick setup, good semantic performance for clause retrieval.
- Small embedding dimensionality is efficient for FAISS.

**Alternative**
- OpenAI `text-embedding-3-small` for higher quality but adds cost + external dependency.

---

## Vector store
**Choice**: **FAISS IndexFlatIP** (cosine similarity) stored in memory.

**Why**
- For 500 PDFs, vectors are typically in tens of thousands → FAISS is fast and simple.
- Minimal ops overhead.

**Alternative**
- For 50k+ docs: Pinecone/Milvus (distributed), plus incremental updates.

---

## Retrieval strategy
**MVP**: dense top-k retrieval (k=5) with score-based confidence.

**Optional v2 improvements**
- Hybrid retrieval: BM25 + dense, union then re-rank.
- Re-ranker: cross-encoder (e.g., `bge-reranker-base`) to improve precision@3.

---

## Hallucination mitigation
**Implemented**:
1) **Refusal gate**: if best similarity score < threshold → refuse.
2) **Strict generation rules**: answer only from context, cite doc/page.
3) **Confidence score**: derived from retrieval score (0–1).

**Why**
- Simple, deterministic, easy to test.
- Prevents “answering from prior knowledge” when documents don’t contain evidence.

---

## Evaluation harness (precision@3)
- 10 manual question-answer pairs for your sample PDFs.
- Metric: whether a correct supporting chunk appears in top 3 retrieved results.

---

## Scaling to 50,000 documents (what changes)
**Bottlenecks**
- Ingestion/embedding time becomes large → distributed embedding workers.
- FAISS on a single machine becomes memory-bound → use ANN index (HNSW/IVF-PQ) or managed vector DB.
- Updates become frequent → incremental indexing required.
- Re-ranking with LLM becomes expensive → use cross-encoder reranker.

**Concrete changes**
- Store embeddings in Pinecone/Milvus.
- Use ANN (HNSW) + metadata filters (doc type/vendor/date).
- Add caching (Redis) and query normalization.
- Use asynchronous retrieval + generation, stream responses.
