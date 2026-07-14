"""
ARIA / Hermes — LLM Client
Supports two backends:
  - local: Qwen2.5-VL-7B-Instruct via HuggingFace Transformers (GPU)
  - openai: External model server with two separate endpoints:
      /v1/chat/completions  → text-only requests
      /v1/multimodal        → requests with image attachments

All agents share one singleton instance.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
import torch
from config import settings

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HermesLLM:
    """
    Singleton LLM client.

    When llm_provider == "openai":
      - Text-only requests  → POST {llm_api_base}{llm_chat_path}
        (default: /v1/chat/completions)
      - Requests with images → POST {llm_api_base}{llm_multimodal_path}
        (default: /v1/multimodal)
      Images are base64-encoded and sent as "image_url" content blocks.

    When llm_provider == "local":
      - Transformers GPU inference (Qwen2.5-VL singleton).
    """

    _instance: HermesLLM | None = None
    _initialized: bool = False

    def __new__(cls) -> HermesLLM:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return

        if settings.llm_provider == "openai":
            self._initialized = True
            logger.info(
                f"✅ LLM → external API  base={settings.llm_api_base}  "
                f"chat={settings.llm_chat_path}  multimodal={settings.llm_multimodal_path}"
            )
            return

        logger.info(f"Loading local LLM: {settings.llm_model_id}")
        torch_dtype = _DTYPE_MAP.get(settings.llm_torch_dtype, torch.bfloat16)

        quant_config = None
        if settings.llm_load_in_4bit and settings.llm_device == "cuda":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            settings.llm_model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if settings.llm_device == "cuda" else settings.llm_device,
            quantization_config=quant_config,
            cache_dir=settings.llm_cache_dir,
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            settings.llm_model_id,
            cache_dir=settings.llm_cache_dir,
            trust_remote_code=True,
        )
        self._initialized = True
        logger.info("✅ Local LLM loaded successfully")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _api_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.llm_api_key and settings.llm_api_key != "not-needed":
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        return headers

    def _encode_image(self, img: bytes | str | Path) -> str:
        """Return a base64-encoded JPEG/PNG data-URI string."""
        if isinstance(img, (str, Path)):
            data = Path(img).read_bytes()
        else:
            data = img
        return base64.b64encode(data).decode("utf-8")

    def _build_api_messages(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Inject image content blocks into the last user message.
        Returns (formatted_messages, has_images).
        """
        if not images:
            return messages, False

        formatted = [dict(m) for m in messages]

        # Find last user message
        last_user = None
        for m in reversed(formatted):
            if m["role"] == "user":
                last_user = m
                break

        if last_user:
            original = last_user["content"]
            content = (
                list(original)
                if isinstance(original, list)
                else [{"type": "text", "text": str(original)}]
            )
            for img in images:
                b64 = self._encode_image(img)
                # Detect MIME from first bytes
                mime = "image/jpeg"
                raw = img if isinstance(img, bytes) else Path(img).read_bytes()
                if raw[:4] == b"\x89PNG":
                    mime = "image/png"
                elif raw[:4] == b"RIFF":
                    mime = "image/webp"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            last_user["content"] = content

        return formatted, True

    def _api_endpoint(self, has_images: bool) -> str:
        """Choose the correct endpoint path based on whether images are included."""
        base = settings.llm_api_base.rstrip("/")
        path = settings.llm_multimodal_path if has_images else settings.llm_chat_path
        return f"{base}{path}"

    # ── API Generation ────────────────────────────────────────────────────────

    async def _generate_api(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
        max_new_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        msgs, has_images = self._build_api_messages(messages, images)
        url = self._api_endpoint(has_images)
        logger.debug(f"LLM API → {url} (images={has_images})")

        payload: dict[str, Any] = {
            "model": settings.llm_model_id,
            "messages": msgs,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": settings.llm_top_p,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=self._api_headers())
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _stream_api(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
        max_new_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        msgs, has_images = self._build_api_messages(messages, images)
        url = self._api_endpoint(has_images)
        logger.debug(f"LLM stream → {url} (images={has_images})")

        payload: dict[str, Any] = {
            "model": settings.llm_model_id,
            "messages": msgs,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": settings.llm_top_p,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload, headers=self._api_headers()) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            pass

    # ── Local Generation ──────────────────────────────────────────────────────

    def _build_local_inputs(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _generate_local_sync(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
        max_new_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        inputs = self._build_local_inputs(messages, images)
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": settings.llm_top_p,
            "do_sample": temperature > 0 and not json_mode,
        }
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

    def _run_generation_thread(self, gen_kwargs: dict[str, Any]) -> None:
        with torch.inference_mode():
            self.model.generate(**gen_kwargs)

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Non-streaming generation.
        Routes to /v1/chat/completions (text) or /v1/multimodal (images).
        """
        _max = max_new_tokens or settings.llm_max_new_tokens
        _temp = temperature or settings.llm_temperature

        if settings.llm_provider == "openai":
            return await self._generate_api(messages, images, _max, _temp, json_mode)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_local_sync, messages, images, _max, _temp, json_mode,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation.
        Routes to /v1/chat/completions (text) or /v1/multimodal (images).
        """
        _max = max_new_tokens or settings.llm_max_new_tokens
        _temp = temperature or settings.llm_temperature

        if settings.llm_provider == "openai":
            async for token in self._stream_api(messages, images, _max, _temp):
                yield token
            return

        # Local streaming
        inputs = self._build_local_inputs(messages, images)
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": _max,
            "temperature": _temp,
            "top_p": settings.llm_top_p,
            "do_sample": _temp > 0,
        }
        thread = Thread(target=self._run_generation_thread, args=(gen_kwargs,))
        thread.start()
        for token in streamer:
            if token:
                yield token
            await asyncio.sleep(0)
        thread.join()

    # ── Convenience ───────────────────────────────────────────────────────────

    async def chat(self, system: str, user: str, **kwargs: Any) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return await self.generate(messages, **kwargs)

    async def json_chat(self, system: str, user: str, **kwargs: Any) -> str:
        return await self.chat(system, user, json_mode=True, **kwargs)

    async def stream_generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Alias for agent compatibility."""
        async for token in self.stream(messages, max_new_tokens=max_new_tokens, temperature=temperature):
            yield token

    @classmethod
    def get(cls) -> HermesLLM:
        return cls()


llm = HermesLLM()


def get_llm() -> HermesLLM:
    return llm
