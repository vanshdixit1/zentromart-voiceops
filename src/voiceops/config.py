from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    dataset: Path
    gemini_api_key: str
    gemini_fallback_api_key: str
    gemini_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    gemini_timeout: float
    gemini_min_request_interval: float
    embedding_timeout: float
    gnani_key: str
    stt_language: str
    tts_language: str
    tts_model: str
    tts_voice: str


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return Settings(
        dataset=Path(os.getenv("VOICEOPS_DATASET", "dataset/Capstone_Project_Dataset_FTTT_2026")),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_fallback_api_key=os.getenv("GEMINI_FALLBACK_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        embedding_api_key=os.getenv("OPENAI_API_KEY", ""),
        embedding_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        gemini_timeout=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60")),
        gemini_min_request_interval=float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "0")),
        embedding_timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
        gnani_key=os.getenv("GNANI_API_KEY", ""),
        stt_language=os.getenv("GNANI_STT_LANGUAGE", "en-IN"),
        tts_language=os.getenv("GNANI_TTS_LANGUAGE", "en-IN"),
        tts_model=os.getenv("GNANI_TTS_MODEL", "timbre-v2.5"),
        tts_voice=os.getenv("GNANI_TTS_VOICE", "Kaveri"),
    )


def resolve_dataset(path: Path) -> Path:
    nested = path / "Capstone_Project_Dataset_FTTT_2026"
    return nested if not (path / "README.md").exists() and (nested / "README.md").exists() else path
