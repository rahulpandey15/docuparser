from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Swap implementations via DOCUPARSER_EMBEDDING_PROVIDER. Nothing else may import concrete classes."""

    #: stable identifier, used in the Chroma collection name
    name: str

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
