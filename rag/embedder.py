"""
ARIA / Hermes — Embedding Module
Uses nomic-ai/nomic-embed-text-v1.5 for all vector embeddings.
Single GPU-accelerated instance shared across the app.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class HermesEmbedder:
    """
    Singleton embedding model using nomic-embed-text-v1.5.
    768-dimensional embeddings, GPU-accelerated.
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

        logger.info(f"Loading embedding model: {settings.embed_model_id}")
        self.model = SentenceTransformer(
            settings.embed_model_id,
            trust_remote_code=True,  # required for nomic-embed
            cache_folder=settings.llm_cache_dir,
        )
        self.model = self.model.to(settings.embed_device)
        self._initialized = True
        logger.info("✅ Embedding model loaded")

    def embed(self, texts: list[str] | str, prompt_name: str = "search_document") -> list[list[float]]:
        """
        Embed one or more texts.

        nomic-embed-text-v1.5 uses task-specific prefixes:
        - "search_document": for embedding documents to be retrieved
        - "search_query": for embedding search queries
        - "clustering": for clustering tasks
        - "classification": for classification tasks
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=settings.embed_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # for cosine similarity
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (uses search_query prompt)."""
        result = self.embed([query], prompt_name="search_query")
        return result[0]

    def embed_document(self, text: str) -> list[float]:
        """Embed a document for storage."""
        result = self.embed([text], prompt_name="search_document")
        return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return self.embed(texts, prompt_name="search_document")

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    @classmethod
    def get(cls) -> HermesEmbedder:
        return cls()


# Module-level singleton
embedder = HermesEmbedder()


def get_embedder() -> HermesEmbedder:
    """FastAPI dependency."""
    return embedder
