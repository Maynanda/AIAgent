"""
ARIA / Hermes — LLM Client
Singleton wrapper around Qwen2.5-VL-7B-Instruct via HuggingFace Transformers.
Loaded once at startup, shared across all agents via dependency injection.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from threading import Thread
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

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
    Singleton LLM client for Qwen2.5-VL-7B-Instruct.

    Supports:
    - Text generation (all agents)
    - Vision input (images in emails, screenshots)
    - Streaming output (WebSocket token-by-token)
    - Structured JSON output (for tool calls)
    """

    _instance: HermesLLM | None = None
    _initialized: bool = False

    def __new__(cls) -> HermesLLM:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        """Load model and processor. Call once at startup."""
        if self._initialized:
            return

        logger.info(f"Loading LLM: {settings.llm_model_id}")
        logger.info(f"Device: {settings.llm_device} | dtype: {settings.llm_torch_dtype} | 4-bit: {settings.llm_load_in_4bit}")

        torch_dtype = _DTYPE_MAP.get(settings.llm_torch_dtype, torch.bfloat16)

        # Build quantization config (4-bit NF4 — fits 7B in ~5GB VRAM)
        quant_config = None
        if settings.llm_load_in_4bit and settings.llm_device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

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
        logger.info("✅ LLM loaded successfully")

    # ── Core generation ──────────────────────────────────────────────────────

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
        # Move to model device
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
        """Non-streaming generation — returns full response string."""
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
            # Bias toward JSON structure — basic approach
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens
        generated = output[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation — yields tokens one by one.
        Uses TextIteratorStreamer in a background thread.
        """
        inputs = self._build_inputs(messages, images)

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

        # Run generation in background thread (non-blocking)
        thread = Thread(target=self._run_generation_thread, args=(gen_kwargs,))
        thread.start()

        # Yield tokens as they are produced
        loop = asyncio.get_event_loop()
        for token in streamer:
            if token:
                yield token
            # Yield control back to event loop between tokens
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
