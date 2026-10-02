"""Composition root: the only place that wires concrete providers together."""
from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .embeddings.factory import get_embedder
from .generation.llm import get_llm
from .ingestion.image_describers import get_captioner
from .ingestion.pipeline import IngestionPipeline
from .retrieval.retriever import Retriever
from .service import RAGService
from .vectorstore.factory import get_vector_store


def build_service(s: Settings | None = None) -> RAGService:
    s = s or get_settings()
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder(s)
    store = get_vector_store(s, embedder)
    ingestion = IngestionPipeline(s, embedder, store, get_captioner(s))
    return RAGService(s, ingestion, Retriever(s, embedder, store), store, get_llm(s))


@lru_cache
def get_service() -> RAGService:
    return build_service()
