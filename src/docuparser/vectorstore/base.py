from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], k: int,
              where: dict[str, Any] | None = None) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]: ...
