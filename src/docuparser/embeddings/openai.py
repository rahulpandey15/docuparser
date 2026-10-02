from __future__ import annotations

from openai import OpenAI

from .base import Embedder

_KNOWN_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
               "text-embedding-ada-002": 1536}


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, api_key: str | None, base_url: str | None = None) -> None:
        self.name = f"openai-{model}"
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._dim = _KNOWN_DIMS.get(model) or len(self.embed_query("dimension probe"))

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            resp = self._client.embeddings.create(model=self._model, input=texts[i:i + 96])
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
