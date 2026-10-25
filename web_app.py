from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
sys.path.insert(0, str(ROOT / "src"))

from voiceops.agent import AgentResponse, SupportAgent
from voiceops.config import Settings, load_settings, resolve_dataset
from voiceops.data import StoreData
from voiceops.llm import GeminiChatModel, LLMError
from voiceops.retrieval import configured_knowledge
from voiceops.tools import AgentTools
from voiceops.voice import GnaniVoice, VoiceError

MAX_JSON_BYTES = 4 * 1024 * 1024 if os.getenv("VERCEL") else 16 * 1024 * 1024
MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
MAX_SESSIONS = 100


@dataclass
class _Conversation:
    agent: SupportAgent
    step: int = 3
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_used: float = field(default_factory=time.monotonic)


class VoiceOpsWebService:
    """Web transport that delegates all support logic to the existing agent."""

    def __init__(self, settings: Settings | None = None, *, model=None, knowledge=None,
                 tools=None, voice=None):
        self.settings = settings or load_settings()
        dataset_path = self.settings.dataset
        if not dataset_path.is_absolute():
            dataset_path = ROOT / dataset_path
        self.dataset = resolve_dataset(dataset_path.resolve())
        if not (self.dataset / "README.md").is_file():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset}. Extract the capstone ZIP under the project's dataset folder."
            )

        self.model = model or GeminiChatModel(
            self.settings.gemini_api_key, self.settings.gemini_model,
            self.settings.gemini_timeout, self.settings.gemini_min_request_interval,
            self.settings.gemini_fallback_api_key,
        )
        self.knowledge = knowledge or configured_knowledge(self.dataset, self.settings)
        self.tools = tools or AgentTools(StoreData(self.dataset))
        self._voice = voice
        self._voice_lock = threading.Lock()
        self._conversation_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions: dict[str, _Conversation] = {}
        session_key = os.getenv("VOICEOPS_SESSION_KEY", "").strip()
        self._session_cipher = Fernet(session_key.encode("ascii")) if session_key else None

    def new_session(self, step: int = 3) -> str:
        if step not in (1, 2, 3):
            raise ValueError("Choose Step 1, Step 2, or Step 3.")
        token = secrets.token_urlsafe(32)
        with self._sessions_lock:
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].last_used)
                del self._sessions[oldest]
            self._sessions[token] = _Conversation(
                SupportAgent(self.model, step, self.knowledge, self.tools), step=step
            )
        return token

    def _session(self, token: str) -> _Conversation:
        with self._sessions_lock:
            session = self._sessions.get(token)
            if session is None:
                raise ValueError("Conversation expired. Start a new conversation and try again.")
            session.last_used = time.monotonic()
            return session

    def export_session(self, token: str) -> str:
        """Encrypt resumable conversation state for stateless serverless requests."""
        if self._session_cipher is None:
            return ""
        session = self._session(token)
        state = {
            "session_id": token,
            "expires_at": int(time.time()) + 3600,
            "step": session.step,
            "history": session.agent.history,
            "verified_orders": session.agent.verified_orders,
            "verification_failures": session.agent.verification_failures,
            "pending_verification_orders": sorted(session.agent.pending_verification_orders),
            "escalated_refund_orders": sorted(session.agent.escalated_refund_orders),
        }
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self._session_cipher.encrypt(payload).decode("ascii")

    def restore_session(self, token: str, encrypted_state: str) -> None:
        """Restore signed, encrypted state supplied by the browser between function calls."""
        if self._session_cipher is None:
            self._session(token)
            return
        if not encrypted_state or len(encrypted_state) > 256_000:
            raise ValueError("Conversation expired. Start a new conversation and try again.")
        try:
            state = json.loads(self._session_cipher.decrypt(encrypted_state.encode("ascii"), ttl=3600))
        except (InvalidToken, UnicodeEncodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("Conversation expired. Start a new conversation and try again.") from exc
        if (not isinstance(state, dict) or state.get("session_id") != token
                or int(state.get("expires_at", 0)) < int(time.time())
                or state.get("step") not in (1, 2, 3)):
            raise ValueError("Conversation expired. Start a new conversation and try again.")
        agent = SupportAgent(self.model, state["step"], self.knowledge, self.tools)
        agent.history = state.get("history", [])
        agent.verified_orders = state.get("verified_orders", {})
        agent.verification_failures = state.get("verification_failures", {})
        agent.pending_verification_orders = set(state.get("pending_verification_orders", []))
        agent.escalated_refund_orders = set(state.get("escalated_refund_orders", []))
        with self._sessions_lock:
            self._sessions[token] = _Conversation(agent, step=state["step"])

    def _get_voice(self) -> GnaniVoice:
        if not self.settings.gnani_key and self._voice is None:
            raise VoiceError("Set GNANI_API_KEY in .env before using microphone or speech playback.")
        with self._voice_lock:
            if self._voice is None:
                self._voice = GnaniVoice(
                    self.settings.gnani_key, self.settings.stt_language,
                    self.settings.tts_model, self.settings.tts_voice,
                )
            return self._voice

    @staticmethod
    def _result(response: AgentResponse, transcript: str = "", audio: bytes = b"") -> dict:
        return {
            "transcript": transcript,
            "answer": response.answer,
            "sources": response.sources,
            "retrieved_chunks": response.retrieved_chunks,
            "tool_events": response.tool_events,
            "audio_base64": base64.b64encode(audio).decode("ascii") if audio else "",
        }

    def _respond(self, session: _Conversation, message: str) -> AgentResponse:
        # The shared knowledge index can lazily initialize embeddings; serialize turns
        # so two new browser requests cannot trigger duplicate initialization calls.
        with session.lock, self._conversation_lock:
            return session.agent.respond(message)

    def _speak(self, text: str) -> bytes:
        voice = self._get_voice()
        with tempfile.TemporaryDirectory(prefix="voiceops-web-") as temp:
            output = Path(temp) / "reply.wav"
            with self._voice_lock:
                voice.synthesize(text, output, getattr(self.settings, "tts_language", self.settings.stt_language))
            return output.read_bytes()

    def process_text(self, token: str, message: str, speak: bool = False) -> dict:
        message = message.strip()
        if not message:
            raise ValueError("Type a question before sending it.")
        if len(message) > 4000:
            raise ValueError("Please keep each message under 4,000 characters.")
        session = self._session(token)
        response = self._respond(session, message)
        audio = self._speak(response.answer) if speak else b""
        return self._result(response, audio=audio)

    @staticmethod
    def _validate_wav(audio: bytes) -> None:
        if not audio:
            raise ValueError("The recording is empty. Record a short question and try again.")
        if len(audio) > MAX_AUDIO_BYTES:
            raise ValueError("Recording is too large. Please keep it under one minute and try again.")
        try:
            with wave.open(io.BytesIO(audio), "rb") as stream:
                if stream.getcomptype() != "NONE":
                    raise ValueError("Please send an uncompressed WAV recording.")
                if stream.getnchannels() not in (1, 2) or stream.getsampwidth() not in (1, 2, 3, 4):
                    raise ValueError("The recording format is unsupported. Please record again.")
                if not 8000 <= stream.getframerate() <= 48000:
                    raise ValueError("The recording sample rate is unsupported. Please record again.")
                duration = stream.getnframes() / max(stream.getframerate(), 1)
                if duration <= 0 or duration > MAX_AUDIO_SECONDS:
                    raise ValueError("Please keep microphone recordings to 60 seconds or less.")
        except (wave.Error, EOFError) as exc:
            raise ValueError("The recording could not be read as WAV audio. Please record again.") from exc

    def process_audio(self, token: str, audio: bytes) -> dict:
        self._validate_wav(audio)
        session = self._session(token)
        if session.step != 3:
            raise ValueError("Microphone conversations use Step 3 so order tools and conversation memory stay enabled.")
        voice = self._get_voice()
        with tempfile.TemporaryDirectory(prefix="voiceops-web-") as temp:
            input_path = Path(temp) / "caller.wav"
            output_path = Path(temp) / "reply.wav"
            input_path.write_bytes(audio)
            with session.lock:
                with self._voice_lock:
                    transcript = voice.transcribe(input_path, self.settings.stt_language)
                if transcript.strip():
                    with self._conversation_lock:
                        response = session.agent.respond(transcript.strip())
                else:
                    response = AgentResponse("Sorry, I didn't catch that. Could you say it again?")
                with self._voice_lock:
                    voice.synthesize(response.answer, output_path,
                                     getattr(self.settings, "tts_language", self.settings.stt_language))
            return self._result(response, transcript.strip(), output_path.read_bytes())


def make_handler(service: VoiceOpsWebService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VoiceOpsLocal/1.0"
        sys_version = ""

        def log_message(self, fmt, *args):
            # Keep local transcripts and question text out of the server console log.
            print(f"{self.address_string()} - {fmt % args}")

        def _send(self, status: int, payload: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; connect-src 'self'; media-src 'self' blob:; "
                             "style-src 'self'; script-src 'self'; img-src 'self' data:; "
                             "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, value: dict):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(status, payload, "application/json; charset=utf-8")

        def _body(self) -> dict:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("Invalid request size.") from exc
            if length <= 0 or length > MAX_JSON_BYTES:
                raise ValueError("Request is empty or too large.")
            try:
                value = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Request must contain valid JSON.") from exc
            if not isinstance(value, dict):
                raise ValueError("Request must be a JSON object.")
            return value

        def do_GET(self):
            route = self._route()
            if route == "/api/health":
                self._json(200, {
                    "ok": True,
                    "dataset_ready": (service.dataset / "README.md").is_file(),
                    "voice_ready": bool(service.settings.gnani_key),
                    "model": service.settings.gemini_model,
                    "hosted": bool(os.getenv("VERCEL")),
                })
                return
            if route == "/":
                path = WEB_ROOT / "index.html"
            elif route in {"/assets/app.js", "/assets/style.css"}:
                path = WEB_ROOT / route.removeprefix("/assets/")
            else:
                self._json(404, {"error": "Page not found."})
                return
            try:
                content = path.read_bytes()
            except OSError:
                self._json(500, {"error": "The website files are missing."})
                return
            kind = "text/html; charset=utf-8" if path.suffix == ".html" else (
                "text/javascript; charset=utf-8" if path.suffix == ".js" else "text/css; charset=utf-8")
            self._send(200, content, kind)

        def do_POST(self):
            route = self._route()
            try:
                body = self._body()
                if route == "/api/session":
                    try:
                        step = int(body.get("step", 3))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Choose Step 1, Step 2, or Step 3.") from exc
                    session_id = service.new_session(step)
                    self._json(200, {"session_id": session_id, "step": step,
                                     "session_state": service.export_session(session_id)})
                    return
                token = body.get("session_id")
                if not isinstance(token, str) or not token:
                    raise ValueError("Start a conversation before sending a message.")
                service.restore_session(token, body.get("session_state", ""))
                if route == "/api/chat":
                    message = body.get("message")
                    if not isinstance(message, str):
                        raise ValueError("Message must be text.")
                    result = service.process_text(token, message, bool(body.get("speak", False)))
                elif route == "/api/voice":
                    encoded = body.get("audio_base64")
                    if not isinstance(encoded, str):
                        raise ValueError("A WAV recording is required.")
                    try:
                        audio = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ValueError("Recording data was invalid. Please record again.") from exc
                    result = service.process_audio(token, audio)
                else:
                    self._json(404, {"error": "Endpoint not found."})
                    return
                if service._session_cipher is not None:
                    result["session_state"] = service.export_session(token)
                self._json(200, result)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except (LLMError, VoiceError) as exc:
                self._json(502, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "Something went wrong. Check the PowerShell window for details and try again."})

        def log_error(self, *args):
            # Avoid dumping request payloads, which may contain order IDs or proof values.
            print("Website request failed:", args[0] if args else "unknown error")

        def _route(self):
            parsed = urlparse(self.path)
            if os.getenv("VERCEL"):
                from urllib.parse import parse_qs
                forwarded = parse_qs(parsed.query).get("__voiceops_route", [""])[0]
                if forwarded.startswith(("/",)) and len(forwarded) <= 200:
                    return forwarded
            return parsed.path

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local VoiceOps website")
    parser.add_argument("--host", default="127.0.0.1", help="Local address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Local web port (default: 8000)")
    args = parser.parse_args()
    try:
        service = VoiceOpsWebService()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    except (LLMError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Could not start the VoiceOps website: {exc}", file=sys.stderr)
        return 2
    print(f"VoiceOps website is ready at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop it. The server is local-only by default.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VoiceOps website.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
