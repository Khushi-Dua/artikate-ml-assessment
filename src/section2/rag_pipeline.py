from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    document: str
    page: int
    chunk_id: str
    text: str


@dataclass
class QueryResult:
    answer: str
    sources: List[Dict]
    confidence: float


class PDFIngestor:
    def read_pdf_pages(self, pdf_path: Path) -> List[Tuple[int, str]]:
        reader = PdfReader(str(pdf_path))
        pages: List[Tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append((i, txt))
        return pages


class Chunker:
    """
    Page-aware sliding window chunking.
    We keep page number on each chunk for citations.
    """

    def __init__(self, chunk_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def chunk_page(self, document: str, page: int, text: str) -> List[Chunk]:
        words = text.split()
        if not words:
            return []

        words_per_chunk = max(50, int(self.chunk_tokens / 1.3))
        words_overlap = max(0, int(self.overlap_tokens / 1.3))
        stride = max(1, words_per_chunk - words_overlap)

        chunks: List[Chunk] = []
        idx = 0
        for start in range(0, len(words), stride):
            window = words[start : start + words_per_chunk]
            if len(window) < 20:
                break
            chunk_text = " ".join(window)
            chunk_id = f"{document}:p{page}:c{idx}"
            chunks.append(Chunk(document=document, page=page, chunk_id=chunk_id, text=chunk_text))
            idx += 1
        return chunks


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)  # cosine via IP on normalized vectors
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape[0] == len(chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, k: int = 5) -> Tuple[List[Chunk], List[float]]:
        scores, idxs = self.index.search(query_emb.reshape(1, -1), k)
        idx_list = idxs[0].tolist()
        score_list = scores[0].tolist()
        found_chunks = [self.chunks[i] for i in idx_list if i >= 0]
        found_scores = [s for i, s in zip(idx_list, score_list) if i >= 0]
        return found_chunks, found_scores


class RAGPipeline:
    """
    Fully-local RAG pipeline:
    - local PDF parsing
    - local embeddings (sentence-transformers)
    - local FAISS search
    - local "answer": retrieval-only summary with citations (no LLM call)
    """

    def __init__(
        self,
        pdf_dir: str = "data/sample_pdfs",
        embed_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = 5,
        refusal_threshold: float = 0.25,
    ) -> None:
        self.pdf_dir = Path(pdf_dir)
        self.top_k = top_k
        self.refusal_threshold = refusal_threshold

        self.ingestor = PDFIngestor()
        self.chunker = Chunker()
        self.embedder = Embedder(embed_model)

        self.store: Optional[FaissStore] = None

    def build_index(self) -> None:
        pdf_paths = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDFs found under {self.pdf_dir}. Ensure PDFs exist in that folder."
            )

        all_chunks: List[Chunk] = []
        for pdf in pdf_paths:
            doc = pdf.name
            for page_num, page_text in self.ingestor.read_pdf_pages(pdf):
                all_chunks.extend(self.chunker.chunk_page(doc, page_num, page_text))

        embeddings = self.embedder.embed([c.text for c in all_chunks])
        self.store = FaissStore(dim=embeddings.shape[1])
        self.store.add(all_chunks, embeddings)

    def _confidence(self, top_score: float) -> float:
        return float(max(0.0, min(1.0, (top_score + 1) / 2)))

    def query(self, question: str) -> QueryResult:
        if self.store is None:
            self.build_index()

        q_emb = self.embedder.embed([question])[0]
        chunks, scores = self.store.search(q_emb, k=self.top_k)

        if not scores or scores[0] < self.refusal_threshold:
            return QueryResult(
                answer="I don't have sufficient information in the provided documents to answer this accurately.",
                sources=[],
                confidence=0.0,
            )

        # Retrieval-only answer: show best-matching snippets + citations.
        top = chunks[:3]
        bullets = []
        for ch in top:
            snippet = ch.text.strip().replace("\n", " ")
            snippet = (snippet[:350] + "...") if len(snippet) > 350 else snippet
            bullets.append(f"- ({ch.document}, page {ch.page}) {snippet}")

        answer = (
            "Local mode (no OpenAI quota required). Top retrieved evidence:\n"
            + "\n".join(bullets)
            + "\n\nIf you enable an LLM later, you can generate a natural-language answer from these citations."
        )

        sources = [{"document": ch.document, "page": ch.page, "chunk": ch.text} for ch in top]
        conf = self._confidence(scores[0])
        return QueryResult(answer=answer, sources=sources, confidence=conf)
