from __future__ import annotations

from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, provider: str = "openai") -> None:
        self.settings = get_settings()
        self.provider = provider
        self._openai_client = None
        self._gemini_client = None
        self._groq_client = None
        self._init_client()

    _HTTP_TIMEOUT: float = 30.0

    def _init_client(self) -> None:
        if self.provider == "openai":
            api_key = self.settings.OPENAI_API_KEY
            if api_key and api_key != "your_openai_api_key":
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=api_key, timeout=self._HTTP_TIMEOUT)
                log.info("LLMClient initialized with OpenAI provider (timeout=%ss)", self._HTTP_TIMEOUT)
            else:
                log.warning("OPENAI_API_KEY not set — LLMClient unavailable")
        elif self.provider == "gemini":
            api_key = self.settings.GEMINI_API_KEY
            if api_key and api_key != "your_gemini_api_key" and api_key.strip():
                from google import genai
                self._gemini_client = genai.Client(api_key=api_key)
                log.info("LLMClient initialized with Gemini provider")
            else:
                log.warning("GEMINI_API_KEY not set — LLMClient unavailable")
        elif self.provider == "groq":
            api_key = self.settings.GROQ_API_KEY
            if api_key and api_key != "your_groq_api_key" and api_key.strip():
                from openai import OpenAI
                self._groq_client = OpenAI(
                    api_key=api_key,
                    base_url=self.settings.GROQ_BASE_URL,
                    timeout=self._HTTP_TIMEOUT,
                )
                log.info("LLMClient initialized with Groq provider (timeout=%ss)", self._HTTP_TIMEOUT)
            else:
                log.warning("GROQ_API_KEY not set — LLMClient unavailable")
        else:
            log.warning("Unknown LLM provider: %s", self.provider)

    @property
    def is_available(self) -> bool:
        return self._openai_client is not None or self._gemini_client is not None or self._groq_client is not None

    def embed(self, texts: list[str]) -> list[Optional[list[float]]]:
        if not texts:
            return []
        if self.provider == "openai" and self._openai_client:
            return self._embed_openai(texts)
        if self.provider == "gemini" and self._gemini_client:
            return self._embed_gemini(texts)
        if self.provider == "groq" and self._groq_client:
            return self._embed_groq(texts)
        log.warning("Embedding unavailable — no client configured")
        return [None] * len(texts)

    def generate_chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        if self.provider == "openai" and self._openai_client:
            return self._generate_openai(messages, temperature, response_format)
        if self.provider == "gemini" and self._gemini_client:
            return self._generate_gemini(messages, temperature, response_format)
        if self.provider == "groq" and self._groq_client:
            return self._generate_groq(messages, temperature, response_format)
        log.warning("Generation unavailable — no client configured")
        return None

    # ── OpenAI ──────────────────────────────────────────────────

    _EMBEDDING_MAX_TOKENS_PER_BATCH: int = 250_000

    def _embed_batched(
        self, texts: list[str], embed_fn, **kwargs: object
    ) -> list[list[float] | None]:
        """Split texts into token-safe batches and concatenate results."""
        batch_size = max(1, self._EMBEDDING_MAX_TOKENS_PER_BATCH // self.settings.RAG_CHUNK_SIZE)
        results: list[list[float] | None] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = embed_fn(input=batch, **kwargs)
                results.extend([r.embedding for r in resp.data])
            except Exception as exc:
                log.error("Embedding batch %d/%d failed: %s", i // batch_size + 1, (len(texts) + batch_size - 1) // batch_size, exc)
                results.extend([None] * len(batch))
        return results

    def _embed_openai(self, texts: list[str]) -> list[list[float] | None]:
        return self._embed_batched(
            texts,
            self._openai_client.embeddings.create,
            model=self.settings.RAG_EMBEDDING_MODEL,
        )

    def _generate_openai(
        self,
        messages: list[dict],
        temperature: float,
        response_format: Optional[dict],
    ) -> Optional[str]:
        kwargs = {
            "model": self.settings.RAG_GENERATION_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self._openai_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as exc:
            log.error("OpenAI generation failed: %s", exc)
            return None

    # ── Gemini ──────────────────────────────────────────────────

    def _embed_gemini(self, texts: list[str]) -> list[Optional[list[float]]]:
        try:
            from google.genai import types
            result = self._gemini_client.models.embed_content(
                model=self.settings.GEMINI_EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=1536,
                ),
            )
            return [e.values for e in result.embeddings]
        except Exception as exc:
            for t in texts:
                log.error("Gemini embedding failed on text (first 120 chars): %r ...", t[:120])
                break
            return [None] * len(texts)

    def _generate_gemini(
        self,
        messages: list[dict],
        temperature: float,
        response_format: Optional[dict],
    ) -> Optional[str]:
        try:
            from google.genai import types

            system_prompt = None
            user_contents: list[str] = []
            for m in messages:
                if m["role"] == "system":
                    system_prompt = m["content"]
                elif m["role"] == "user":
                    user_contents.append(m["content"])

            config_kwargs: dict = {
                "temperature": temperature,
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if response_format and response_format.get("type") == "json_object":
                config_kwargs["response_mime_type"] = "application/json"

            config = types.GenerateContentConfig(**config_kwargs)
            content = "\n".join(user_contents) if user_contents else ""

            resp = self._gemini_client.models.generate_content(
                model=self.settings.GEMINI_GENERATION_MODEL,
                contents=content,
                config=config,
            )
            return resp.text
        except Exception as exc:
            log.error("Gemini generation failed: %s", exc)
            return None

    # ── Groq (OpenAI-compatible) ───────────────────────────────

    def _embed_groq(self, texts: list[str]) -> list[list[float] | None]:
        return self._embed_batched(
            texts,
            self._groq_client.embeddings.create,
            model=self.settings.GROQ_EMBEDDING_MODEL,
        )

    def _generate_groq(
        self,
        messages: list[dict],
        temperature: float,
        response_format: Optional[dict],
    ) -> Optional[str]:
        kwargs = {
            "model": self.settings.GROQ_GENERATION_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self._groq_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as exc:
            log.error("Groq generation failed: %s", exc)
            return None
