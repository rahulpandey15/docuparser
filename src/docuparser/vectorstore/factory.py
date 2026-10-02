from __future__ import annotations

from ..config import Settings
from ..embeddings.base import Embedder
from .base import VectorStore


def get_vector_store(s: Settings, embedder: Embedder) -> VectorStore:
    from .chroma import ChromaStore
    return ChromaStore(s, embedder.name, embedder.dimension)
