from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..embeddings.base import Embedder
from ..types import Chunk
from ..vectorstore.base import VectorStore
from .chunking import split_text
from .image_describers import Captioner
from .pdf_parser import parse_pdf

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    source: str
    pages_with_content: int
    text_chunks: int
    image_chunks: int


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


class IngestionPipeline:
    def __init__(self, s: Settings, embedder: Embedder, store: VectorStore,
                 captioner: Captioner | None) -> None:
        self.s, self.embedder, self.store, self.captioner = s, embedder, store, captioner

    def ingest_pdf(self, path: Path, source_name: str | None = None) -> IngestResult:
        path = Path(path)
        source = source_name or path.name
        doc_id = file_hash(path)  # content hash => re-ingesting the same file is idempotent
        segments = parse_pdf(path, self.s, self.captioner)

        chunks: list[Chunk] = []
        for seg in segments:
            for i, piece in enumerate(split_text(seg.text, self.s.chunk_size,
                                                 self.s.chunk_overlap)):
                meta = {"doc_id": doc_id, "source": source, "page": seg.page,
                        "chunk_type": seg.kind, **seg.extra}
                chunks.append(Chunk(f"{doc_id}:{len(chunks)}", piece, meta))

        self.store.delete_document(doc_id)  # drop stale chunks if chunking params changed
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.store.add(chunks, vectors)

        n_img = sum(c.metadata["chunk_type"] in ("image", "ocr_page") for c in chunks)
        log.info("Ingested %s: %d chunks (%d image-derived)", source, len(chunks), n_img)
        return IngestResult(doc_id, source, len({c.metadata["page"] for c in chunks}),
                            len(chunks) - n_img, n_img)
