from __future__ import annotations

from .base import Embedder


class HuggingFaceEmbedder(Embedder):
    def __init__(self, model: str, device: str = "cpu", query_prefix: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install extras: uv sync --extra hf") from e
        self.name = f"hf-{model}"
        self._model = SentenceTransformer(model, device=device)
        self._prefix = query_prefix
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=32, normalize_embeddings=True,
                                  show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(self._prefix + text, normalize_embeddings=True)
        return vec.tolist()
