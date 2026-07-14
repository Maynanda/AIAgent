"""
ARIA / Hermes — LLM Client
Singleton wrapper around Qwen2.5-VL-7B-Instruct via HuggingFace Transformers or external OpenAI-compatible API.
Loaded once at startup, shared across all agents via dependency injection.
"""
from __future__ import annotations

import asyncio
import logging
import base64
import json
from pathlib import Path
from collections.abc import AsyncGenerator
from threading import Thread
from typing import Any

import torch
import httpx
from config import settings

logger = logging.getLogger(__name__)

# ── dtype map ───────────────────────────────────────────────────────────────
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HermesLLM:
    """
    Singleton LLM client.
    Can be run:
    - Localy via Transformers (loaded directly on GPU).
    - Remotely via OpenAI-compatible completion APIs (layer 0 model server).

    Supports:
    - Text generation
    - Vision input (images in user messages / attachments)
    - Streaming responses (WebSocket support)
    - Structured JSON output
    """

    _instance: HermesLLM | None = None
    _initialized: bool = False

    def __new__(cls) -> HermesLLM:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        """Load model and processor or initialize API connection. Call once at startup."""
        if self._initialized:
            return

        if settings.llm_provider == "openai":
            self._initialized = True
            logger.info(f"✅ LLM configured to use external API: {settings.llm_api_base} with model {settings.llm_model_id}")
            return

        logger.info(f"Loading local LLM: {settings.llm_model_id}")
        logger.info(f"Device: {settings.llm_device} | dtype: {settings.llm_torch_dtype} | 4-bit: {settings.llm_load_in_4bit}")

        torch_dtype = _DTYPE_MAP.get(settings.llm_torch_dtype, torch.bfloat16)

        # Build quantization config (4-bit NF4 — fits 7B in ~5GB VRAM)
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

    # ── API Generation ────────────────────────────────────────────────────────

    def _format_messages_for_api(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Format messages and optional images into OpenAI image_url structures."""
        if not images:
            return messages

        formatted_messages = []
        for msg in messages:
            formatted_messages.append(dict(msg))

        last_user_msg = None
        for msg in reversed(formatted_messages):
            if msg["role"] == "user":
                last_user_msg = msg
                break

        if last_user_msg:
            original_content = last_user_msg["content"]
            content_list = []
            if isinstance(original_content, list):
                content_list = list(original_content)
            else:
                content_list = [{"type": "text", "text": str(original_content)}]

            for img in images:
                if isinstance(img, (str, Path)):
                    img_path = Path(img)
                    if img_path.exists():
                        mime = "image/jpeg"
                        if img_path.suffix.lower() in [".png"]:
                            mime = "image/png"
                        elif img_path.suffix.lower() in [".webp"]:
                            mime = "image/webp"
                        
                        with open(img_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encoded_string}"
                            }
                        })
                elif isinstance(img, bytes):
                    encoded_string = base64.b64encode(img).decode("utf-8")
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}"
                        }
                    })

            last_user_msg["content"] = content_list

        return formatted_messages

    async def _generate_api(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        payload_messages = self._format_messages_for_api(messages, images)
        payload = {
            "model": settings.llm_model_id,
            "messages": payload_messages,
            "max_tokens": max_new_tokens or settings.llm_max_new_tokens,
            "temperature": temperature or settings.llm_temperature,
            "top_p": settings.llm_top_p,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {}
        if settings.llm_api_key and settings.llm_api_key != "not-needed":
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.llm_api_base.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()

    async def _stream_api(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        payload_messages = self._format_messages_for_api(messages, images)
        payload = {
            "model": settings.llm_model_id,
            "messages": payload_messages,
            "max_tokens": max_new_tokens or settings.llm_max_new_tokens,
            "temperature": temperature or settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "stream": True,
        }

        headers = {}
        if settings.llm_api_key and settings.llm_api_key != "not-needed":
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{settings.llm_api_base.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            pass

    # ── Local Generation ──────────────────────────────────────────────────────

    def _build_inputs(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Convert messages + optional images to model inputs."""
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    async def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Text generation — routes to local or API client."""
        if settings.llm_provider == "openai":
            return await self._generate_api(messages, images, max_new_tokens, temperature, json_mode)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            messages,
            images,
            max_new_tokens or settings.llm_max_new_tokens,
            temperature or settings.llm_temperature,
            json_mode,
        )

    def _generate_sync(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
        max_new_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        inputs = self._build_inputs(messages, images)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": settings.llm_top_p,
            "do_sample": temperature > 0,
        }
        if json_mode:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated = output[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation — routes to local or API client."""
        if settings.llm_provider == "openai":
            async for chunk in self._stream_api(messages, images, max_new_tokens, temperature):
                yield chunk
            return

        inputs = self._build_inputs(messages, images)

        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens or settings.llm_max_new_tokens,
            "temperature": temperature or settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "do_sample": (temperature or settings.llm_temperature) > 0,
        }

        thread = Thread(target=self._run_generation_thread, args=(gen_kwargs,))
        thread.start()

        loop = asyncio.get_event_loop()
        for token in streamer:
            if token:
                yield token
            await asyncio.sleep(0)

        thread.join()

    def _run_generation_thread(self, gen_kwargs: dict[str, Any]) -> None:
        with torch.inference_mode():
            self.model.generate(**gen_kwargs)

    # ── Convenience helpers ──────────────────────────────────────────────────

    async def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """Simple text-only chat with system + user message."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return await self.generate(messages, **kwargs)

    async def json_chat(self, system: str, user: str, **kwargs: Any) -> str:
        """Chat with JSON output mode enabled."""
        return await self.chat(system, user, json_mode=True, **kwargs)

    # ── Streaming chat integration helper (Agent compatibility) ──────────────

    async def stream_generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Alias helper for agent message templates streaming."""
        async for token in self.stream(messages, max_new_tokens=max_new_tokens, temperature=temperature):
            yield token

    # ── Singleton accessor ───────────────────────────────────────────────────

    @classmethod
    def get(cls) -> HermesLLM:
        """Get the singleton instance (must call initialize() first)."""
        return cls()


# Module-level singleton
llm = HermesLLM()


def get_llm() -> HermesLLM:
    """FastAPI dependency — returns the initialized LLM singleton."""
    return llm
