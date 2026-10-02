from __future__ import annotations

import logging
from pathlib import Path

import typer

from .container import build_service

app = typer.Typer(help="RAG over PDFs (text + images) with ChromaDB")
logging.basicConfig(level=logging.INFO)


@app.command()
def ingest(path: Path = typer.Argument(..., exists=True, help="PDF file or folder of PDFs")):
    """Ingest one PDF or every PDF in a folder."""
    svc = build_service()
    files = sorted(path.glob("**/*.pdf")) if path.is_dir() else [path]
    for f in files:
        r = svc.ingest(f)
        typer.echo(f"{r.source}: {r.text_chunks} text + {r.image_chunks} image chunks "
                   f"(doc_id={r.doc_id})")


@app.command()
def ask(question: str, top_k: int = typer.Option(None, "--top-k", "-k"),
        show_sources: bool = typer.Option(True, "--sources/--no-sources")):
    """Ask a question about the indexed documents."""
    ans = build_service().ask(question, top_k)
    if ans.answer:
        typer.echo(f"\n{ans.answer}\n")
    if show_sources or ans.answer is None:
        for i, h in enumerate(ans.sources, 1):
            m = h.chunk.metadata
            typer.echo(f"[{i}] {m['source']} p.{m['page']} ({m['chunk_type']}) "
                       f"score={h.score:.3f}\n    {h.chunk.text[:200]!r}")


@app.command("list")
def list_docs():
    """List indexed documents."""
    for d in build_service().list_documents():
        typer.echo(f"{d['doc_id']}  {d['source']}  chunks={d['chunks']} "
                   f"(image-derived={d['image_chunks']})")


@app.command()
def delete(doc_id: str):
    """Delete a document from the index."""
    build_service().delete_document(doc_id)
    typer.echo(f"Deleted {doc_id}")


if __name__ == "__main__":
    app()
