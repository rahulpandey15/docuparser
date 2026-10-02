"""Deterministic hashing embedder. For tests and offline smoke runs only (no semantic quality)."""
from __future__ import annotations

import hashlib
import math
import re

from .base import Embedder


class FakeEmbedder(Embedder):
    name = "fake-hash-256"

    @property
    def dimension(self) -> int:
        return 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 256
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
