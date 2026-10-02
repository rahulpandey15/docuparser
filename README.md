# DocuParser: PDF text + images RAG with ChromaDB

Ask questions over PDFs. Text is extracted per page; images inside PDFs are converted to
searchable text (caption via a vision model and/or OCR) and indexed alongside the text, so
answers can cite figures too.

## Architecture

```
PDF -> parse (PyMuPDF) -> [text | image -> caption/OCR] -> chunk -> embed -> ChromaDB
Question -> embed -> ChromaDB top-k -> prompt with numbered sources -> LLM -> answer + citations
```

| Layer | Module | Swap via |
|---|---|---|
| Embeddings | `embeddings/` (HuggingFace, OpenAI) | `DOCUPARSER_EMBEDDING_PROVIDER` |
| Vector store | `vectorstore/` (Chroma) | add adapter + edit `factory.py` |
| Image -> text | `ingestion/image_describers.py` | `DOCUPARSER_IMAGE_MODE`, `DOCUPARSER_CAPTION_PROVIDER` |
| LLM | `generation/llm.py` (OpenAI-compatible, HF local, none) | `DOCUPARSER_LLM_PROVIDER` |
| Wiring | `container.py` (single composition root) | |

Each embedding model gets its **own Chroma collection** (name derived from the model), so
switching HuggingFace -> OpenAI never mixes incompatible vectors. After switching, run
`uv run python scripts/reindex.py` to re-embed the PDFs in `data/raw/`.

## Quick start (uv)

```bash
uv sync --extra hf --extra ocr      # creates .venv from uv.lock; drop extras you don't need
cp .env.example .env                # set DOCUPARSER_OPENAI_API_KEY (or Ollama / llm_provider=none)

uv run docuparser ingest ./my_pdfs  # file or folder
uv run docuparser ask "What does the revenue chart show?"
uv run docuparser list

make run                            # API at http://localhost:8000/docs
```

Add a dependency with `uv add <pkg>` (or `uv add --optional hf <pkg>`, `uv add --dev <pkg>`).
Plain pip users: `uv export --no-dev --extra hf --extra ocr -o requirements.txt` then `pip install -r requirements.txt && pip install -e .`.

API: `POST /ingest` (multipart PDF), `POST /query` `{question, top_k?, doc_id?}`,
`GET /documents`, `DELETE /documents/{doc_id}`, `GET /health`.

Docker: `docker compose up --build` (API + Chroma server).

## Modes worth knowing

- **No LLM key?** `DOCUPARSER_LLM_PROVIDER=none` returns retrieved passages only. For a local LLM
  use Ollama: `DOCUPARSER_OPENAI_BASE_URL=http://localhost:11434/v1`, `DOCUPARSER_LLM_MODEL=llama3.1`.
- **Images:** `DOCUPARSER_IMAGE_MODE=none|ocr|caption|both`. `ocr` also OCRs scanned (text-less) pages
  and needs the Tesseract binary. `caption` with `huggingface` runs BLIP locally (generic
  captions); `openai` uses a vision model (much better for charts/diagrams).
- **Offline smoke test:** `DOCUPARSER_EMBEDDING_PROVIDER=fake` (hash vectors, no model download).

## Tests

`uv run pytest -q` (uses the fake embedder/captioner, so no downloads or API keys needed).

## Known limitations / optimization ideas

Tables are extracted as plain text (consider `pymupdf4llm` or `unstructured`); no hybrid
(BM25) search or reranker yet; ingestion is synchronous (move to a queue for large files);
no auth on the API; direct CLIP image embeddings are not implemented (caption-then-embed is).
