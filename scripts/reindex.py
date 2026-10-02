"""Re-ingest every PDF in data/raw into the collection for the *currently configured* embedder.

Use after changing DOCUPARSER_EMBEDDING_PROVIDER / DOCUPARSER_EMBEDDING_MODEL (e.g. HuggingFace -> OpenAI).
Each embedding model has its own Chroma collection, so the old one stays intact until you delete it.
"""
import logging

from docuparser.container import build_service
from docuparser.config import get_settings

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    s = get_settings()
    svc = build_service(s)
    for pdf in sorted(s.raw_dir.glob("*.pdf")):
        r = svc.ingest(pdf)
        print(f"{r.source}: {r.text_chunks}+{r.image_chunks} chunks")
