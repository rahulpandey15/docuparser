from __future__ import annotations

import hashlib
import re
from typing import Any

import chromadb

from ..config import Settings
from ..types import Chunk, RetrievedChunk
from .base import VectorStore


def collection_name(prefix: str, embedder_name: str) -> str:
    """One collection per embedding model: vectors from different models are not comparable."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", embedder_name).strip("-").lower()
    h = hashlib.md5(embedder_name.encode()).hexdigest()[:6]
    return f"{prefix}-{slug[:40]}-{h}".strip("-")[:63]


class ChromaStore(VectorStore):
    def __init__(self, s: Settings, embedder_name: str, dimension: int) -> None:
        if s.chroma_mode == "http":
            self._client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)
        else:
            s.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(s.chroma_path))
        self._col = self._client.get_or_create_collection(
            name=collection_name(s.collection_prefix, embedder_name),
            metadata={"hnsw:space": "cosine", "embedding_model": embedder_name,
                      "dimension": dimension},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), 500):  # stay under Chroma's max batch size
            part = chunks[i:i + 500]
            self._col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                metadatas=[c.metadata for c in part],
                embeddings=embeddings[i:i + 500],
            )

    def query(self, embedding: list[float], k: int,
              where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=min(k, n),
                              where=where or None,
                              include=["documents", "metadatas", "distances"])
        out = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append(RetrievedChunk(Chunk(id_, doc, meta), score=1.0 - float(dist)))
        return out

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def list_documents(self) -> list[dict[str, Any]]:
        data = self._col.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for m in data["metadatas"]:
            d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"],
                                              "chunks": 0, "image_chunks": 0})
            d["chunks"] += 1
            d["image_chunks"] += int(m.get("chunk_type") in ("image", "ocr_page"))
        return sorted(docs.values(), key=lambda d: d["source"])
