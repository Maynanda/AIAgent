"""
ARIA / Hermes — Embedding Module
Supports two backends:
  - local: nomic-ai/nomic-embed-text-v1.5 via sentence-transformers (GPU)
  - api: External embedding server with two endpoints:
      /v1/embeddings        → single text or small batch
      /v1/embeddings/batch  → large batch processing
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class HermesEmbedder:
    """
    Singleton embedding client.

    When embed_provider == "api":
      Single embed  → POST {embed_api_base}{embed_api_path}
        body: {"model": "...", "input": "text or [texts]"}
      Batch embed   → POST {embed_api_base}{embed_batch_api_path}
        body: {"model": "...", "inputs": ["text1", "text2", ...]}

    When embed_provider == "local":
      Uses SentenceTransformer (nomic-embed-text-v1.5) on GPU.
    """

    _instance: HermesEmbedder | None = None
    _initialized: bool = False

    def __new__(cls) -> HermesEmbedder:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return

        if settings.embed_provider == "api":
            self._initialized = True
            logger.info(
                f"✅ Embedder → external API  base={settings.embed_api_base}  "
                f"single={settings.embed_api_path}  batch={settings.embed_batch_api_path}"
            )
            return

        logger.info(f"Loading local embedding model: {settings.embed_model_id}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            settings.embed_model_id,
            trust_remote_code=True,
            cache_folder=settings.llm_cache_dir,
        )
        self.model = self.model.to(settings.embed_device)
        self._initialized = True
        logger.info("✅ Embedding model loaded")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _api_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.embed_api_key and settings.embed_api_key != "not-needed":
            headers["Authorization"] = f"Bearer {settings.embed_api_key}"
        return headers

    def _single_url(self) -> str:
        return settings.embed_api_base.rstrip("/") + settings.embed_api_path

    def _batch_url(self) -> str:
        return settings.embed_api_base.rstrip("/") + settings.embed_batch_api_path

    # ── API Embedding ─────────────────────────────────────────────────────────

    async def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """
        Calls /v1/embeddings for 1 text or short lists,
        /v1/embeddings/batch for larger sets.
        """
        headers = self._api_headers()

        if len(texts) == 1:
            # Single endpoint: {input: "text"}
            url = self._single_url()
            payload = {"model": settings.embed_model_id, "input": texts[0]}
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                # Support both {embedding: [...]} and OpenAI-style {data: [{embedding: [...]}]}
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                if "embedding" in data:
                    return [data["embedding"]]
                raise ValueError(f"Unexpected embedding API response format: {list(data.keys())}")

        else:
            # Batch endpoint: {inputs: ["text1", "text2", ...]}
            url = self._batch_url()
            payload = {"model": settings.embed_model_id, "inputs": texts}
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                # Support {embeddings: [[...]]} or {data: [{embedding: [...]}]}
                if "embeddings" in data:
                    return data["embeddings"]
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                raise ValueError(f"Unexpected batch embedding API response format: {list(data.keys())}")

    # ── Sync API wrappers (for code that calls embed_* synchronously) ─────────

    def _run_async(self, coro) -> Any:
        """Run an async coroutine synchronously (for non-async callers)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use a thread executor instead
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ── Local Embedding ───────────────────────────────────────────────────────

    def _embed_local(self, texts: list[str], prompt_name: str = "search_document") -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=settings.embed_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    # ── Public Interface ──────────────────────────────────────────────────────

    def embed(self, texts: list[str] | str, prompt_name: str = "search_document") -> list[list[float]]:
        """Embed one or more texts synchronously. Routes to API or local model."""
        if isinstance(texts, str):
            texts = [texts]

        if settings.embed_provider == "api":
            return self._run_async(self._embed_api(texts))

        return self._embed_local(texts, prompt_name)

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        if settings.embed_provider == "api":
            results = self._run_async(self._embed_api([query]))
            return results[0]
        return self._embed_local([query], prompt_name="search_query")[0]

    def embed_document(self, text: str) -> list[float]:
        """Embed a document for storage."""
        if settings.embed_provider == "api":
            results = self._run_async(self._embed_api([text]))
            return results[0]
        return self._embed_local([text], prompt_name="search_document")[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (uses batch endpoint if API mode)."""
        if settings.embed_provider == "api":
            return self._run_async(self._embed_api(texts))
        return self._embed_local(texts, prompt_name="search_document")

    # ── Async versions (preferred in async contexts) ──────────────────────────

    async def async_embed(self, texts: list[str] | str) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if settings.embed_provider == "api":
            return await self._embed_api(texts)
        return self._embed_local(texts)

    async def async_embed_query(self, query: str) -> list[float]:
        results = await self.async_embed([query])
        return results[0]

    async def async_embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.async_embed(texts)

    # ── Utility ───────────────────────────────────────────────────────────────

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    @classmethod
    def get(cls) -> HermesEmbedder:
        return cls()


embedder = HermesEmbedder()


def get_embedder() -> HermesEmbedder:
    return embedder
