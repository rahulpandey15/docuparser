from __future__ import annotations

from pathlib import Path

from .config import Settings
from .generation.llm import LLM
from .generation.prompts import SYSTEM_PROMPT, build_user_prompt
from .ingestion.pipeline import IngestionPipeline, IngestResult
from .retrieval.retriever import Retriever
from .types import Answer
from .vectorstore.base import VectorStore


class RAGService:
    def __init__(self, s: Settings, ingestion: IngestionPipeline, retriever: Retriever,
                 store: VectorStore, llm: LLM | None) -> None:
        self.s, self.ingestion, self.retriever, self.store, self.llm = (
            s, ingestion, retriever, store, llm)

    def ingest(self, path: Path, source_name: str | None = None) -> IngestResult:
        return self.ingestion.ingest_pdf(path, source_name)

    def ask(self, question: str, top_k: int | None = None, doc_id: str | None = None) -> Answer:
        hits = self.retriever.retrieve(question, top_k, doc_id)
        if not hits:
            return Answer(question, "No relevant content found in the indexed documents.", [])
        if self.llm is None:  # retrieval-only mode
            return Answer(question, None, hits)
        answer = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits))
        return Answer(question, answer, hits)

    def list_documents(self):
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
