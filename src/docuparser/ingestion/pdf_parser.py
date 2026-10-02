from __future__ import annotations

import io
import logging
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from PIL import Image

from ..config import Settings
from ..types import Segment
from .image_describers import Captioner, ocr_image

log = logging.getLogger(__name__)


def _describe(img: Image.Image, mode: str, captioner: Captioner | None) -> str:
    parts = []
    if mode in ("caption", "both") and captioner:
        try:
            cap = captioner.caption(img)
            if cap:
                parts.append(f"Image description: {cap}")
        except Exception as e:
            log.warning("Captioning failed: %s", e)
    if mode in ("ocr", "both"):
        txt = ocr_image(img)
        if len(txt) > 10:
            parts.append(f"Text in image: {txt}")
    return "\n".join(parts)


def parse_pdf(path: Path, s: Settings, captioner: Captioner | None) -> list[Segment]:
    """Extract per-page text plus (optionally) descriptions of embedded images."""
    segments: list[Segment] = []
    seen_xrefs: set[int] = set()
    with fitz.open(path) as doc:
        for page in doc:
            pno = page.number + 1
            text = page.get_text("text").strip()

            if len(text) < 20 and s.ocr_scanned_pages and s.image_mode in ("ocr", "both"):
                # scanned page without a text layer -> OCR the full page render
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = ocr_image(img)
                if ocr_text:
                    segments.append(Segment(ocr_text, pno, "ocr_page"))
            elif text:
                segments.append(Segment(text, pno, "text"))

            if s.image_mode == "none":
                continue
            for n, info in enumerate(page.get_images(full=True)):
                xref = info[0]
                if xref in seen_xrefs:  # same logo/figure repeated across pages
                    continue
                seen_xrefs.add(xref)
                try:
                    raw = doc.extract_image(xref)
                    img = Image.open(io.BytesIO(raw["image"]))
                    img.load()
                except Exception as e:
                    log.warning("Skipping unreadable image (page %s): %s", pno, e)
                    continue
                if min(img.size) < s.min_image_side:
                    continue
                desc = _describe(img, s.image_mode, captioner)
                if desc:
                    segments.append(Segment(f"[Figure on page {pno}]\n{desc}", pno, "image",
                                            {"image_index": n, "width": img.width,
                                             "height": img.height}))
    return segments
