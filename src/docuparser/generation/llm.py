from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Settings


class LLM(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


class OpenAICompatLLM(LLM):
    """Works with OpenAI, Azure-compatible gateways, Ollama, vLLM, LM Studio (set base_url)."""

    def __init__(self, s: Settings) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=s.openai_api_key or "none",
                              base_url=s.openai_base_url or None)
        self._s = s

    def generate(self, system: str, user: str) -> str:
        r = self._client.chat.completions.create(
            model=self._s.llm_model, temperature=self._s.llm_temperature,
            max_tokens=self._s.llm_max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()


class HFLocalLLM(LLM):
    def __init__(self, s: Settings) -> None:
        try:
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install extras: uv sync --extra hf") from e
        self._pipe = pipeline("text-generation", model=s.llm_model,
                              device=-1 if s.embedding_device == "cpu" else s.embedding_device)
        self._s = s

    def generate(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = self._pipe(msgs, max_new_tokens=self._s.llm_max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()


def get_llm(s: Settings) -> LLM | None:
    if s.llm_provider == "none":
        return None
    if s.llm_provider == "openai":
        return OpenAICompatLLM(s)
    if s.llm_provider == "huggingface":
        return HFLocalLLM(s)
    raise ValueError(f"Unknown LLM provider: {s.llm_provider}")
