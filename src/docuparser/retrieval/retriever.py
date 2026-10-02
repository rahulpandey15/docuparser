from __future__ import annotations

from ..config import Settings
from ..embeddings.base import Embedder
from ..types import RetrievedChunk
from ..vectorstore.base import VectorStore


class Retriever:
    def __init__(self, s: Settings, embedder: Embedder, store: VectorStore) -> None:
        self.s, self.embedder, self.store = s, embedder, store

    def retrieve(self, question: str, top_k: int | None = None,
                 doc_id: str | None = None) -> list[RetrievedChunk]:
        k = top_k or self.s.top_k
        where = {"doc_id": doc_id} if doc_id else None
        hits = self.store.query(self.embedder.embed_query(question), k, where)
        return [h for h in hits if h.score >= self.s.min_score]
