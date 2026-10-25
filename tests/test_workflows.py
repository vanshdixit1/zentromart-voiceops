from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
TEST_TMP = ROOT / "tests" / ".tmp"
TEST_TMP.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import batch_run
import evaluate
from voiceops.agent import SupportAgent
from voiceops.data import StoreData, read_csv
from voiceops.llm import GeminiChatModel
from voiceops.retrieval import KnowledgeBase
from voiceops.tools import AgentTools


class OfflineModel:
    def complete(self, messages, **kwargs):
        return "I'm ZentroMart's AI assistant. I can help with your question."


class FakeVoice:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path, language=None):
        return f"Test caller asks about policy in {Path(audio_path).stem}."

    def synthesize(self, text, output_path, language="en-IN"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF-test-audio")
        return output_path


class WorkflowTests(unittest.TestCase):
    def test_audio_duration_supports_gnani_streaming_wav_headers(self):
        import wave

        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            audio_path = Path(temp) / "gnani.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 8000)
            raw = bytearray(audio_path.read_bytes())
            data_chunk = raw.index(b"data")
            raw[4:8] = b"\xff\xff\xff\xff"
            raw[data_chunk + 4:data_chunk + 8] = b"\xff\xff\xff\xff"
            audio_path.write_bytes(raw)
            self.assertEqual(batch_run.audio_duration_seconds(audio_path), 0.5)

    def test_gemini_responses_preserve_order_verification_tool_flow(self):
        from unittest.mock import Mock
        from google import genai

        customers = {row["customer_id"]: row for row in read_csv(DATASET / "data" / "customers.csv")}
        order = read_csv(DATASET / "data" / "orders.csv")[0]
        proof = customers[order["customer_id"]]["phone"][-4:]
        responses = [
            types.SimpleNamespace(text=json.dumps({
                "action": "lookup_order", "order_id": order["order_id"], "verification_value": "",
            })),
            types.SimpleNamespace(text=json.dumps({
                "action": "lookup_order", "order_id": order["order_id"], "verification_value": proof,
            })),
            types.SimpleNamespace(text=f"Your verified order status is {order['status']}; your digits were {proof}."),
        ]
        fake_client = types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=Mock(side_effect=responses))
        )
        with patch.object(genai, "Client", return_value=fake_client):
            model = GeminiChatModel("unit-test-key", "gemini-3.5-flash-lite")
            agent = SupportAgent(model, 3, tools=AgentTools(StoreData(DATASET)))
            first = agent.respond(f"Where is order {order['order_id']}?")
            verified = agent.respond(proof)

        self.assertIn("delivery pincode", first.answer)
        self.assertTrue(verified.tool_events[0]["verified"])
        self.assertNotIn(proof, verified.answer)
        self.assertNotIn(proof, str(agent.history))
        # The order lookup and proof are deterministic, so only the follow-up
        # customer question should need a Gemini generation request.
        self.assertEqual(fake_client.models.generate_content.call_count, 1)

    def test_evaluator_runs_practice_questions_and_conversations_offline(self):
        settings = types.SimpleNamespace(
            gemini_api_key="offline-test", gemini_fallback_api_key="", gemini_model="offline-test",
            embedding_api_key="", embedding_base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small", gemini_timeout=1,
            gemini_min_request_interval=0, embedding_timeout=1, dataset=DATASET,
        )
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            output = Path(temp) / "evaluation.jsonl"
            argv = ["evaluate.py", "--dataset", str(DATASET), "--steps", "1", "2", "3",
                    "--output", str(output)]
            stdout = io.StringIO()
            with patch.object(sys, "argv", argv), \
                 patch.object(evaluate, "load_settings", return_value=settings), \
                 patch.object(evaluate, "GeminiChatModel", return_value=OfflineModel()), \
                 patch.object(evaluate, "configured_knowledge",
                              side_effect=lambda dataset, cfg: KnowledgeBase(dataset)), \
                 contextlib.redirect_stdout(stdout):
                self.assertEqual(evaluate.main(), 0)

            summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["step_1_questions"]["cases"], 52)
            self.assertEqual(summary["step_2_questions"]["cases"], 52)
            self.assertEqual(summary["step_3_questions"]["cases"], 52)
            self.assertEqual(summary["step_3_conversations"]["cases"], 96)
            first_question = read_csv(DATASET / "practice" / "questions.csv")[0]
            first_saved = next(json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                               if json.loads(line).get("kind") == "question"
                               and json.loads(line).get("step") == 1)
            self.assertEqual(first_saved["input"], first_question["asr_text"])
            self.assertIn("Step 1 (asr_text)", stdout.getvalue())
            self.assertIn("Step 2 (asr_text)", stdout.getvalue())

    def test_evaluator_limit_caps_cases_per_step(self):
        settings = types.SimpleNamespace(
            gemini_api_key="offline-test", gemini_fallback_api_key="", gemini_model="offline-test",
            embedding_api_key="", embedding_base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small", gemini_timeout=1,
            gemini_min_request_interval=0, embedding_timeout=1, dataset=DATASET,
        )
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            output = Path(temp) / "evaluation.jsonl"
            argv = ["evaluate.py", "--dataset", str(DATASET), "--steps", "1", "2", "3",
                    "--limit", "3", "--output", str(output)]
            with patch.object(sys, "argv", argv), \
                 patch.object(evaluate, "load_settings", return_value=settings), \
                 patch.object(evaluate, "GeminiChatModel", return_value=OfflineModel()), \
                 patch.object(evaluate, "configured_knowledge",
                              side_effect=lambda dataset, cfg: KnowledgeBase(dataset)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(), 0)

            summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["step_1_questions"]["cases"], 3)
            self.assertEqual(summary["step_2_questions"]["cases"], 3)
            self.assertEqual(summary["step_3_questions"]["cases"], 3)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            question_rows = [row for row in rows if row["kind"] == "question"]
            conversation_rows = [row for row in rows if row["kind"] == "conversation"]
            all_conversations = read_csv(DATASET / "practice" / "conversations.csv")
            first_conversation_ids = list(dict.fromkeys(
                row["conversation_id"] for row in all_conversations))[:3]
            self.assertEqual(len(question_rows), 9)
            self.assertEqual({row["conversation_id"] for row in conversation_rows},
                             set(first_conversation_ids))
            self.assertEqual(len(conversation_rows), sum(
                row["conversation_id"] in set(first_conversation_ids) for row in all_conversations))
            self.assertEqual({row["input"] for row in rows if row["step"] == 1},
                             {row["asr_text"] for row in read_csv(DATASET / "practice" / "questions.csv")[:3]})

    def test_evaluator_resume_skips_saved_cases_without_duplicate_rows(self):
        settings = types.SimpleNamespace(
            gemini_api_key="offline-test", gemini_fallback_api_key="", gemini_model="offline-test",
            embedding_api_key="", embedding_base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small", gemini_timeout=1,
            gemini_min_request_interval=0, embedding_timeout=1, dataset=DATASET,
        )
        question = read_csv(DATASET / "practice" / "questions.csv")[0]
        practice_id = question.get("id", "")
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            output = Path(temp) / "evaluation.jsonl"
            output.write_text(json.dumps({"kind": "question", "step": 1,
                                          "practice_id": practice_id, "answer": "saved"}) + "\n",
                              encoding="utf-8")
            argv = ["evaluate.py", "--dataset", str(DATASET), "--steps", "1",
                    "--output", str(output), "--resume"]
            with patch.object(sys, "argv", argv), \
                 patch.object(evaluate, "load_settings", return_value=settings), \
                 patch.object(evaluate, "GeminiChatModel", return_value=OfflineModel()), \
                 patch.object(evaluate, "configured_knowledge",
                              side_effect=lambda dataset, cfg: KnowledgeBase(dataset)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(), 0)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 52)
            self.assertEqual(sum(row.get("practice_id") == practice_id for row in rows), 1)
            summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["step_1_questions"]["cases"], 52)

    def test_evaluator_resume_replays_incomplete_conversation_from_turn_one(self):
        settings = types.SimpleNamespace(
            gemini_api_key="offline-test", gemini_fallback_api_key="", gemini_model="offline-test",
            embedding_api_key="", embedding_base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small", gemini_timeout=1,
            gemini_min_request_interval=0, embedding_timeout=1, dataset=DATASET,
        )
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            output = Path(temp) / "evaluation.jsonl"
            output.write_text(json.dumps({"kind": "conversation", "step": 3,
                                          "conversation_id": "PC002", "turn": "1",
                                          "answer": "incomplete"}) + "\n", encoding="utf-8")
            argv = ["evaluate.py", "--dataset", str(DATASET), "--steps", "3",
                    "--output", str(output), "--resume"]
            with patch.object(sys, "argv", argv), \
                 patch.object(evaluate, "load_settings", return_value=settings), \
                 patch.object(evaluate, "GeminiChatModel", return_value=OfflineModel()), \
                 patch.object(evaluate, "configured_knowledge",
                              side_effect=lambda dataset, cfg: KnowledgeBase(dataset)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(), 0)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            conversations = [row for row in rows if row.get("kind") == "conversation"]
            self.assertEqual(len(conversations), 96)
            self.assertEqual(sum(row.get("conversation_id") == "PC002" and row.get("turn") == "1"
                                 for row in conversations), 1)
            self.assertNotEqual(next(row for row in conversations
                                     if row.get("conversation_id") == "PC002" and row.get("turn") == "1"
                                     )["answer"], "incomplete")

    def test_audio_batch_saves_transcript_answer_sources_and_speech(self):
        settings = types.SimpleNamespace(
            gemini_api_key="offline-test", gemini_fallback_api_key="", gemini_model="offline-test",
            embedding_api_key="", embedding_base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small", gemini_timeout=1,
            gemini_min_request_interval=0, embedding_timeout=1, dataset=DATASET,
            gnani_key="offline-test", stt_language="en-IN",
            tts_model="timbre-v2.5", tts_voice="Kaveri",
        )
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            root = Path(temp)
            audio_folder = root / "customer_audio" / "caller_1"
            audio_folder.mkdir(parents=True)
            import wave
            with wave.open(str(audio_folder / "01.wav"), "wb") as audio:
                audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(8000)
                audio.writeframes(b"\0\0" * 800)
            output = root / "sample_outputs"
            argv = ["batch_run.py", str(root / "customer_audio"), "--dataset", str(DATASET),
                    "--output", str(output), "--session-mode", "per-folder"]
            with patch.object(sys, "argv", argv), \
                 patch.object(batch_run, "load_settings", return_value=settings), \
                 patch.object(batch_run, "GnaniVoice", FakeVoice), \
                 patch.object(batch_run, "GeminiChatModel", return_value=OfflineModel()), \
                 patch.object(batch_run, "configured_knowledge",
                              side_effect=lambda dataset, cfg: KnowledgeBase(dataset)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch_run.main(), 0)

            records = [json.loads(line) for line in (output / "batch_results.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertIn("01", records[0]["transcript"])
            self.assertTrue(records[0]["answer"])
            self.assertTrue(records[0]["retrieved_sources"])
            self.assertTrue(records[0]["retrieved_chunks"])
            self.assertEqual(Path(records[0]["output_audio"]).read_bytes(), b"RIFF-test-audio")

    def test_audio_batch_rejects_over_60_seconds_without_calling_gnani(self):
        settings = types.SimpleNamespace(
            gemini_api_key="offline-test", gemini_fallback_api_key="", gemini_model="offline-test",
            embedding_api_key="", embedding_base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small", gemini_timeout=1,
            gemini_min_request_interval=0, embedding_timeout=1, dataset=DATASET,
            gnani_key="offline-test", stt_language="en-IN", tts_model="timbre-v2.5",
            tts_voice="Kaveri",
        )
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            root = Path(temp); audio_folder = root / "audio"; audio_folder.mkdir()
            import wave
            with wave.open(str(audio_folder / "long.wav"), "wb") as audio:
                audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(8000)
                audio.writeframes(b"\0\0" * (8000 * 61))
            self.assertGreater(batch_run.audio_duration_seconds(audio_folder / "long.wav"), 60)
            output = root / "out"
            argv = ["batch_run.py", str(audio_folder), "--dataset", str(DATASET), "--output", str(output)]
            class MustNotTranscribe(FakeVoice):
                def transcribe(self, *args, **kwargs):
                    raise AssertionError("overlong recording must be rejected before STT")
            with patch.object(sys, "argv", argv), \
                 patch.object(batch_run, "load_settings", return_value=settings), \
                 patch.object(batch_run, "GnaniVoice", MustNotTranscribe), \
                 patch.object(batch_run, "GeminiChatModel", return_value=OfflineModel()), \
                 patch.object(batch_run, "configured_knowledge",
                              side_effect=lambda dataset, cfg: KnowledgeBase(dataset)), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(batch_run.main(), 0)
            row = json.loads((output / "batch_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("60 seconds", row["error"])
            self.assertEqual(row["transcript"], "")
            self.assertEqual(row["output_audio"], "")

    def test_compressed_audio_duration_is_read_from_metadata(self):
        mutagen = types.ModuleType("mutagen")
        mutagen.File = lambda path: types.SimpleNamespace(info=types.SimpleNamespace(length=72.5))
        with patch.dict(sys.modules, {"mutagen": mutagen}):
            self.assertEqual(batch_run.audio_duration_seconds(Path("caller.mp3")), 72.5)


if __name__ == "__main__":
    unittest.main()
