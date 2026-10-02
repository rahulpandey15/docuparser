from __future__ import annotations

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    sep, rest = seps[0], seps[1:]
    pieces = text.split(sep) if sep else list(text)
    out: list[str] = []
    for p in pieces:
        if len(p) > size and rest:
            out.extend(_split(p, size, rest))
        else:
            out.append(p)
    return out


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Recursive character splitter: prefers paragraph > line > sentence > word boundaries."""
    text = text.strip()
    if not text:
        return []
    atoms = [a for a in _split(text, chunk_size, _SEPARATORS) if a.strip()]
    chunks: list[str] = []
    cur = ""
    for a in atoms:
        joiner = "\n" if cur else ""
        if len(cur) + len(joiner) + len(a) <= chunk_size:
            cur = f"{cur}{joiner}{a}"
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]  # start overlap on a word boundary
            cur = f"{tail}\n{a}" if tail and len(tail) + len(a) + 1 <= chunk_size else a
    if cur:
        chunks.append(cur)
    return chunks
