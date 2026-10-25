from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from voiceops.agent import SupportAgent
from voiceops.config import load_settings, resolve_dataset
from voiceops.data import StoreData
from voiceops.llm import GeminiChatModel, LLMError
from voiceops.retrieval import configured_knowledge
from voiceops.tools import AgentTools
from voiceops.voice import GnaniVoice, VoiceError

EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_AUDIO_SECONDS = 60


def _wav_data_chunk(path: Path) -> tuple[int, int] | None:
    """Return the WAV data chunk's declared and physically available byte counts."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        offset = 12
        while offset + 8 <= file_size:
            handle.seek(offset)
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                break
            chunk_name = chunk_header[:4]
            declared_size = int.from_bytes(chunk_header[4:], "little")
            data_offset = offset + 8
            if chunk_name == b"data":
                return declared_size, max(0, file_size - data_offset)
            offset = data_offset + declared_size + (declared_size & 1)
    return None


def audio_duration_seconds(path: Path) -> float:
    try:
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as audio:
                rate = max(audio.getframerate(), 1)
                frames = audio.getnframes()
                chunk = _wav_data_chunk(path)
                if chunk and chunk[0] == 0xFFFFFFFF:
                    frame_bytes = audio.getnchannels() * audio.getsampwidth()
                    if frame_bytes < 1:
                        raise VoiceError(f"Cannot validate WAV frame size for {path.name}.")
                    frames = chunk[1] // frame_bytes
                return frames / rate
        from mutagen import File as MutagenFile
        audio = MutagenFile(path)
        duration = getattr(getattr(audio, "info", None), "length", None)
        if duration is None:
            raise ValueError("audio duration is unavailable")
        return float(duration)
    except (wave.Error, EOFError, OSError, ValueError, ImportError) as exc:
        raise VoiceError(f"Cannot validate audio duration for {path.name}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe, answer, and speak each audio file using Gnani")
    parser.add_argument("audio_folder", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sample_outputs"))
    parser.add_argument("--language", help="STT language code, for example en-IN or gu-IN")
    parser.add_argument("--tts-language", help="Reply speech language code, for example hi-IN or hi-en (defaults to GNANI_TTS_LANGUAGE)")
    parser.add_argument("--session-mode", choices=("per-file", "per-folder"), default="per-file",
                        help="Treat each recording as a new caller, or keep turns together by subfolder")
    args = parser.parse_args()
    files = sorted(p for p in args.audio_folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        print(f"No supported audio files found in {args.audio_folder}", file=sys.stderr); return 2
    cfg = load_settings(); dataset = resolve_dataset((args.dataset or cfg.dataset).resolve())
    tts_language = args.tts_language or getattr(cfg, "tts_language", "en-IN")
    voice = GnaniVoice(cfg.gnani_key, cfg.stt_language, cfg.tts_model, cfg.tts_voice)
    model = GeminiChatModel(cfg.gemini_api_key, cfg.gemini_model, cfg.gemini_timeout,
                            cfg.gemini_min_request_interval, cfg.gemini_fallback_api_key)
    kb, tools = configured_knowledge(dataset, cfg), AgentTools(StoreData(dataset))
    session_agents: dict[Path, SupportAgent] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / "batch_results.jsonl"
    with results.open("w", encoding="utf-8") as output:
        for audio in files:
            relative_name = audio.relative_to(args.audio_folder).with_suffix("").as_posix().replace("/", "_")
            spoken_path = args.output / f"{relative_name}_reply.wav"
            transcript = safe_transcript = answer = ""
            sources: list[str] = []
            chunks: list[dict[str, str]] = []
            events: list[dict] = []
            error = ""
            try:
                duration = audio_duration_seconds(audio)
                if duration <= 0 or duration > MAX_AUDIO_SECONDS:
                    raise VoiceError(f"{audio.name} is {duration:.1f} seconds; Gnani accepts clips up to 60 seconds.")
                transcript = voice.transcribe(audio, args.language)
                if transcript:
                    session = audio.parent if args.session_mode == "per-folder" else audio
                    if session not in session_agents:
                        session_agents[session] = SupportAgent(model, 3, kb, tools)
                    agent = session_agents[session]
                    response = agent.respond(transcript)
                    answer, sources, chunks, events = (response.answer, response.sources,
                                                       response.retrieved_chunks, response.tool_events)
                    safe_transcript = agent.redact(transcript)
                else:
                    answer = "Sorry, I didn't catch that. Could you say it again?"
                voice.synthesize(answer, spoken_path, tts_language)
            except Exception as exc:
                error = str(exc)
                print(f"{audio.name}: skipped/failed: {error}", file=sys.stderr)
            record = {"input_audio": str(audio), "transcript": safe_transcript, "answer": answer,
                      "retrieved_sources": sources, "retrieved_chunks": chunks, "tool_events": events,
                      "output_audio": str(spoken_path) if spoken_path.exists() else "", "error": error}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{audio.name}: {safe_transcript or '[unclear audio]'}\n  {answer or '[no reply]'}"
                  f"\n  Speech: {record['output_audio'] or '[not generated]'}")
    print(f"Batch record: {results}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LLMError, VoiceError, FileNotFoundError, ValueError) as exc:
        print(f"Audio batch could not run: {exc}", file=sys.stderr)
        raise SystemExit(2)
