"""Turn images into searchable text: OCR (text inside the image) and captioning (what it shows)."""
from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod

from PIL import Image

from ..config import Settings

log = logging.getLogger(__name__)


class Captioner(ABC):
    @abstractmethod
    def caption(self, image: Image.Image) -> str: ...


class HFCaptioner(Captioner):
    """BLIP loaded directly (the 'image-to-text' pipeline was removed in transformers 5)."""

    def __init__(self, model: str, device: str = "cpu") -> None:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install extras: uv sync --extra hf") from e
        self._device = device
        self._processor = BlipProcessor.from_pretrained(model)
        self._model = BlipForConditionalGeneration.from_pretrained(model).to(device)
        self._model.eval()

    def caption(self, image: Image.Image) -> str:
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = inputs.to(self._device)
        out = self._model.generate(**inputs, max_new_tokens=60)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()


class OpenAIVisionCaptioner(Captioner):
    def __init__(self, model: str, api_key: str | None, base_url: str | None) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    def caption(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        resp = self._client.chat.completions.create(
            model=self._model, max_tokens=250,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this image from a document in detail. "
                 "If it is a chart/table/diagram, state what it shows and the key values."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        )
        return (resp.choices[0].message.content or "").strip()


class FakeCaptioner(Captioner):
    def caption(self, image: Image.Image) -> str:
        return f"fake caption of a {image.width}x{image.height} picture of a red square"


def get_captioner(s: Settings) -> Captioner | None:
    if s.image_mode not in ("caption", "both"):
        return None
    if s.caption_provider == "huggingface":
        return HFCaptioner(s.caption_model, s.embedding_device)
    if s.caption_provider == "openai":
        return OpenAIVisionCaptioner(s.vision_model, s.openai_api_key, s.openai_base_url)
    if s.caption_provider == "fake":
        return FakeCaptioner()
    raise ValueError(f"Unknown caption provider: {s.caption_provider}")


def ocr_image(image: Image.Image) -> str:
    """OCR via Tesseract. Returns '' (with a warning) if pytesseract/tesseract is unavailable."""
    try:
        import pytesseract
        return pytesseract.image_to_string(image).strip()
    except Exception as e:  # missing package or binary
        log.warning("OCR unavailable (%s). Install tesseract + `uv sync --extra ocr`.", e)
        return ""
