from __future__ import annotations

from ..config import Settings
from .base import Embedder


def get_embedder(s: Settings) -> Embedder:
    p = s.embedding_provider
    if p == "huggingface":
        from .huggingface import HuggingFaceEmbedder
        return HuggingFaceEmbedder(s.embedding_model, s.embedding_device, s.embedding_query_prefix)
    if p == "openai":
        from .openai import OpenAIEmbedder
        return OpenAIEmbedder(s.embedding_model, s.openai_api_key, s.openai_base_url)
    if p == "fake":
        from .fake import FakeEmbedder
        return FakeEmbedder()
    raise ValueError(f"Unknown embedding provider: {p}")
