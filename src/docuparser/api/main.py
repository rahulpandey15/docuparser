from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..container import get_service
from ..service import RAGService

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="DocuParser", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_id: str | None = None


class SourceOut(BaseModel):
    source: str
    page: int
    chunk_type: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str | None
    sources: list[SourceOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...), svc: RAGService = Depends(get_service)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    name = Path(file.filename).name
    dest = svc.s.raw_dir / name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
    shutil.move(tmp.name, dest)
    try:
        return svc.ingest(dest, name).__dict__
    except Exception as e:
        raise HTTPException(422, f"Failed to ingest PDF: {e}") from e


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, svc: RAGService = Depends(get_service)):
    ans = svc.ask(req.question, req.top_k, req.doc_id)
    return QueryResponse(
        question=ans.question, answer=ans.answer,
        sources=[SourceOut(source=h.chunk.metadata["source"], page=h.chunk.metadata["page"],
                           chunk_type=h.chunk.metadata["chunk_type"], score=round(h.score, 4),
                           text=h.chunk.text) for h in ans.sources])


@app.get("/documents")
def documents(svc: RAGService = Depends(get_service)):
    return svc.list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, svc: RAGService = Depends(get_service)):
    svc.delete_document(doc_id)
    return {"deleted": doc_id}
