.PHONY: install test lint run reindex
install:
	uv sync --extra hf --extra ocr
test:
	uv run pytest -q
lint:
	uv run ruff check src tests
run:
	uv run uvicorn docuparser.api.main:app --reload --port 8000
reindex:
	uv run python scripts/reindex.py
