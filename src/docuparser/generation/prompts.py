from __future__ import annotations

from ..types import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the user's documents. "
    "Use ONLY the numbered context passages provided. Cite passages inline like [1], [2]. "
    "Some passages are descriptions or OCR text of figures/images, treat them as evidence too. "
    "If the context does not contain the answer, say you could not find it in the documents."
)


def build_user_prompt(question: str, hits: list[RetrievedChunk]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.chunk.metadata
        blocks.append(f"[{i}] (source: {m['source']}, page {m['page']}, {m['chunk_type']})\n"
                      f"{h.chunk.text}")
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"
