"""
ARIA / Hermes — Whisper Speech-to-Text Service
Loads the local OpenAI Whisper model singleton on demand.
"""
from __future__ import annotations

import logging
import asyncio

import whisper

from config import settings

logger = logging.getLogger(__name__)


class HermesWhisper:
    """
    Singleton class wrapping OpenAI Whisper model for audio transcription.
    Loaded on demand, runs on CUDA GPU if configured.
    """

    _instance: HermesWhisper | None = None
    _initialized: bool = False

    def __new__(cls) -> HermesWhisper:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return

        logger.info(f"Loading Whisper model: {settings.whisper_model_size} on device: {settings.whisper_device}")
        self.model = whisper.load_model(
            settings.whisper_model_size,
            device=settings.whisper_device,
        )
        self._initialized = True
        logger.info("✅ Whisper STT model loaded successfully")

    async def transcribe(self, file_path: str) -> str:
        """Transcribe an audio file path to text."""
        if not self._initialized:
            self.initialize()

        loop = asyncio.get_event_loop()
        # Run transcription block in executor to prevent blocking FastAPI async loop
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            file_path,
        )
        return result

    def _transcribe_sync(self, file_path: str) -> str:
        options = {}
        if settings.whisper_language:
            options["language"] = settings.whisper_language

        res = self.model.transcribe(file_path, **options)
        return res.get("text", "").strip()


# Singleton accessor
whisper_stt = HermesWhisper()


async def transcribe_local_audio(file_path: str) -> str:
    """Convenience helper for transcription routing."""
    return await whisper_stt.transcribe(file_path)
