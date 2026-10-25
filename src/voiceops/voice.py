from __future__ import annotations

import re
from pathlib import Path


class VoiceError(RuntimeError):
    pass


def format_for_speech(text: str) -> str:
    """Remove visual formatting and source labels before sending text to Gnani TTS."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"(?im)^\s*(?:sources?|retrieved sources)\s*:\s*.*$", " ", text)
    text = re.sub(r"\[/?(?:\*\*|`)?[A-Z]{2,8}-\d+(?:\.\d+)?(?:;[^\]]*)?\]", " ", text)
    text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s*(?:[-*•]+|\d+[.)])\s*", "", text)
    text = re.sub(r"[*_`#]+", "", text)
    text = text.replace("|", " ")
    text = re.sub(r"(?<!\d)(\d{4})[ -]?(\d{4})(?!\d)", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


class GnaniVoice:
    """Lazy adapter for the course-approved Gnani Vachana speech SDK only."""
    def __init__(self, api_key: str, stt_language: str = "en-IN", model: str = "timbre-v2.5", voice: str = "Kaveri"):
        if not api_key:
            raise VoiceError("Set GNANI_API_KEY before using audio features.")
        import os
        os.environ["GNANI_API_KEY"] = api_key
        try:
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient
        except ImportError as exc:
            raise VoiceError("Install Gnani's official SDK with: pip install gnani-vachana") from exc
        self.stt, self.tts = GnaniSTTClient(), GnaniTTSClient()
        self.language, self.model, self.voice = stt_language, model, voice

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        result = self.stt.transcribe(str(audio_path), language_code=language or self.language)
        if isinstance(result, dict):
            return str(result.get("transcript", result.get("text", ""))).strip()
        return str(result).strip()

    def synthesize(self, text: str, output_path: Path, language: str = "en-IN") -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio = self.tts.synthesize(format_for_speech(text), voice=self.voice,
                                    model=self.model, language=language)
        output_path.write_bytes(audio)
        return output_path

