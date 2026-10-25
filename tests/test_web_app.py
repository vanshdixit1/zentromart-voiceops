from __future__ import annotations

import base64
import io
import json
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path
from http.server import ThreadingHTTPServer
from cryptography.fernet import Fernet, InvalidToken

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from voiceops.data import StoreData
from voiceops.retrieval import KnowledgeBase
from voiceops.tools import AgentTools
from web_app import VoiceOpsWebService, make_handler


class OfflineModel:
    def __init__(self):
        self.prompts = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages)
        return "I'm ZentroMart's AI assistant. I can help with the return policy."


class FakeVoice:
    def __init__(self, transcript="How long do I have to return an item?"):
        self.transcript = transcript
        self.transcribe_calls = []
        self.synthesize_calls = []

    def transcribe(self, path, language=None):
        self.transcribe_calls.append((Path(path).read_bytes(), language))
        return self.transcript

    def synthesize(self, text, output_path, language="en-IN"):
        self.synthesize_calls.append((text, language))
        Path(output_path).write_bytes(b"RIFF-test-voice-reply")
        return Path(output_path)


def settings():
    return types.SimpleNamespace(
        dataset=DATASET, gemini_api_key="offline-test", gemini_fallback_api_key="",
        gemini_model="offline-test", gemini_timeout=1, gemini_min_request_interval=0,
        embedding_api_key="", embedding_base_url="https://api.openai.com/v1",
        embedding_model="text-embedding-3-small", embedding_timeout=1,
        gnani_key="offline-test", stt_language="en-IN", tts_model="timbre-v2.5",
        tts_voice="Kaveri",
    )


def make_service(model=None, voice=None):
    return VoiceOpsWebService(
        settings(), model=model or OfflineModel(), knowledge=KnowledgeBase(DATASET),
        tools=AgentTools(StoreData(DATASET)), voice=voice or FakeVoice(),
    )


def wav_fixture() -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)
    return payload.getvalue()


class WebAppTests(unittest.TestCase):
    def test_encrypted_session_state_restores_memory_without_exposing_it(self):
        import os
        key = Fernet.generate_key().decode("ascii")
        previous = os.environ.get("VOICEOPS_SESSION_KEY")
        os.environ["VOICEOPS_SESSION_KEY"] = key
        try:
            first_service = make_service()
            session_id = first_service.new_session(step=3)
            first_service._session(session_id).agent.history = [
                {"role": "user", "content": "My order is late."},
                {"role": "assistant", "content": "Please share your order ID."},
            ]
            first_service._session(session_id).agent.pending_verification_orders.add("12345678")
            encrypted = first_service.export_session(session_id)
            self.assertNotIn("My order is late", encrypted)

            second_service = make_service()
            second_service.restore_session(session_id, encrypted)
            restored = second_service._session(session_id).agent
            self.assertEqual(restored.history[-1]["content"], "Please share your order ID.")
            self.assertIn("12345678", restored.pending_verification_orders)
            with self.assertRaisesRegex(ValueError, "Conversation expired"):
                second_service.restore_session(session_id, encrypted[:-4] + "xxxx")
        finally:
            if previous is None:
                os.environ.pop("VOICEOPS_SESSION_KEY", None)
            else:
                os.environ["VOICEOPS_SESSION_KEY"] = previous

    def test_website_keeps_prompt_only_and_grounded_step_modes(self):
        model = OfflineModel()
        service = make_service(model=model)

        step_one = service.process_text(service.new_session(step=1), "How long is the return window?")
        step_two = service.process_text(service.new_session(step=2), "How long is the return window?")

        self.assertEqual(step_one["sources"], [])
        self.assertTrue(step_two["sources"])
        self.assertTrue(step_two["retrieved_chunks"])
        with self.assertRaisesRegex(ValueError, "Step 1, Step 2, or Step 3"):
            service.new_session(step=4)

    def test_text_turn_keeps_existing_agent_sources_and_session_memory(self):
        model = OfflineModel()
        service = make_service(model=model)
        session = service.new_session()

        first = service.process_text(session, "How long do I have to return an item?")
        second = service.process_text(session, "And what if it is electronics?")

        self.assertIn("answer", first)
        self.assertTrue(first["sources"])
        self.assertTrue(first["retrieved_chunks"])
        self.assertEqual(first["audio_base64"], "")
        self.assertIn("How long do I have to return an item?",
                      model.prompts[-1][1]["content"])
        self.assertTrue(second["sources"])

    def test_voice_turn_uses_gnani_adapter_and_returns_transcript_sources_and_playback(self):
        voice = FakeVoice()
        service = make_service(voice=voice)
        session = service.new_session()

        result = service.process_audio(session, wav_fixture())

        self.assertEqual(result["transcript"], "How long do I have to return an item?")
        self.assertTrue(result["answer"])
        self.assertTrue(result["sources"])
        self.assertTrue(result["retrieved_chunks"])
        self.assertEqual(voice.transcribe_calls[0][1], "en-IN")
        self.assertEqual(voice.synthesize_calls[0][1], "en-IN")
        self.assertEqual(base64.b64decode(result["audio_base64"]), b"RIFF-test-voice-reply")

    def test_invalid_audio_is_rejected_before_speech_provider_is_called(self):
        voice = FakeVoice()
        service = make_service(voice=voice)
        session = service.new_session()
        with self.assertRaisesRegex(ValueError, "read as WAV"):
            service.process_audio(session, b"not a wave file")
        self.assertEqual(voice.transcribe_calls, [])
        self.assertEqual(voice.synthesize_calls, [])

    def test_local_http_server_serves_site_and_offline_chat_api(self):
        service = make_service()
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        except PermissionError:
            self.skipTest("This execution environment does not permit local socket binding.")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            try:
                first_response = urllib.request.urlopen(base + "/", timeout=3)
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, PermissionError):
                    self.skipTest("This execution environment does not permit local socket connections.")
                raise
            with first_response as response:
                page = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn("Record a question", page)
            with urllib.request.urlopen(base + "/assets/app.js", timeout=3) as response:
                self.assertIn("getUserMedia", response.read().decode("utf-8"))
            with urllib.request.urlopen(urllib.request.Request(
                    base + "/api/session", data=b"{}", headers={"Content-Type": "application/json"},
                    method="POST"), timeout=3) as response:
                session_id = json.loads(response.read())["session_id"]
            payload = json.dumps({"session_id": session_id,
                                  "message": "How long do I have to return an item?",
                                  "speak": True}).encode()
            with urllib.request.urlopen(urllib.request.Request(
                    base + "/api/chat", data=payload, headers={"Content-Type": "application/json"},
                    method="POST"), timeout=3) as response:
                answer = json.loads(response.read())
            self.assertTrue(answer["answer"])
            self.assertTrue(answer["sources"])
            self.assertEqual(base64.b64decode(answer["audio_base64"]), b"RIFF-test-voice-reply")

            voice_payload = json.dumps({
                "session_id": session_id,
                "audio_base64": base64.b64encode(wav_fixture()).decode("ascii"),
            }).encode()
            with urllib.request.urlopen(urllib.request.Request(
                    base + "/api/voice", data=voice_payload,
                    headers={"Content-Type": "application/json"}, method="POST"), timeout=3) as response:
                voice_result = json.loads(response.read())
            self.assertEqual(voice_result["transcript"], "How long do I have to return an item?")
            self.assertTrue(voice_result["sources"])
            self.assertEqual(base64.b64decode(voice_result["audio_base64"]), b"RIFF-test-voice-reply")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
