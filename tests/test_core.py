from __future__ import annotations

import csv
import json
import os
import types
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_TMP = Path(__file__).resolve().parent / ".tmp"
TEST_TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TEST_TMP)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceops.agent import SupportAgent
from voiceops.config import load_settings
from voiceops.data import StoreData, add_business_days, read_csv, refund_date_context
from voiceops.llm import EmbeddingClient, GeminiChatModel, LLMError
from voiceops.retrieval import KnowledgeBase
from voiceops.tools import AgentTools
from voiceops.voice import GnaniVoice


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(rows)


class MockModel:
    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages)
        return self.answers.pop(0)


class CoreTests(unittest.TestCase):
    def test_settings_keep_gemini_chat_separate_from_optional_openai_embeddings(self):
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "gemini-test-key",
            "GEMINI_FALLBACK_API_KEY": "gemini-fallback-test-key",
            "GEMINI_MODEL": "gemini-custom-test",
            "GEMINI_TIMEOUT_SECONDS": "15",
            "GEMINI_MIN_REQUEST_INTERVAL_SECONDS": "4.1",
            "OPENAI_API_KEY": "",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_TIMEOUT_SECONDS": "30",
        }):
            settings = load_settings()
        self.assertEqual(settings.gemini_api_key, "gemini-test-key")
        self.assertEqual(settings.gemini_fallback_api_key, "gemini-fallback-test-key")
        self.assertEqual(settings.gemini_model, "gemini-custom-test")
        self.assertEqual(settings.embedding_api_key, "")
        self.assertEqual(settings.embedding_model, "text-embedding-3-small")
        self.assertEqual(settings.gemini_timeout, 15)
        self.assertEqual(settings.gemini_min_request_interval, 4.1)
        self.assertEqual(settings.embedding_timeout, 30)

    def test_gemini_requires_an_explicit_api_key(self):
        with self.assertRaisesRegex(LLMError, r"GEMINI_API_KEY in \.env"):
            GeminiChatModel("", "gemini-3.5-flash-lite")

    def test_gemini_connection_errors_use_the_application_error_path(self):
        from unittest.mock import Mock
        from google import genai

        fake_client = types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=Mock(side_effect=OSError("network unavailable")))
        )
        with patch.object(genai, "Client", return_value=fake_client):
            model = GeminiChatModel("unit-test-key", "gemini-3.5-flash-lite")
            with self.assertRaisesRegex(LLMError, r"Gemini request failed \(OSError\): network unavailable"):
                model.complete([{"role": "user", "content": "Hello"}])

    def test_gemini_quota_error_falls_back_once_to_second_project_key(self):
        from unittest.mock import Mock
        from google import genai

        class QuotaError(Exception):
            code = 429

        primary_client = types.SimpleNamespace(models=types.SimpleNamespace(
            generate_content=Mock(side_effect=QuotaError("RESOURCE_EXHAUSTED"))))
        fallback_client = types.SimpleNamespace(models=types.SimpleNamespace(
            generate_content=Mock(return_value=types.SimpleNamespace(text="fallback response"))))
        with patch.object(genai, "Client", side_effect=[primary_client, fallback_client]) as factory:
            model = GeminiChatModel("primary-test-key", "gemini-3.5-flash-lite",
                                    fallback_api_key="fallback-test-key")
            result = model.complete([{"role": "user", "content": "Hello"}])

        self.assertEqual(result, "fallback response")
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args_list[0].kwargs["api_key"], "primary-test-key")
        self.assertEqual(factory.call_args_list[1].kwargs["api_key"], "fallback-test-key")
        primary_client.models.generate_content.assert_called_once()
        fallback_client.models.generate_content.assert_called_once()

    def test_gemini_timeout_does_not_retry_with_fallback_key(self):
        from unittest.mock import Mock
        from google import genai
        from google.genai.errors import ServerError

        primary_client = types.SimpleNamespace(models=types.SimpleNamespace(
            generate_content=Mock(side_effect=ServerError(504, {"error": {"message": "deadline exceeded"}}))))
        fallback_client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=Mock()))
        with patch.object(genai, "Client", side_effect=[primary_client, fallback_client]):
            model = GeminiChatModel("primary-test-key", "gemini-3.5-flash-lite",
                                    fallback_api_key="fallback-test-key")
            with self.assertRaisesRegex(LLMError, "timed out; no automatic retry"):
                model.complete([{"role": "user", "content": "Hello"}])

        primary_client.models.generate_content.assert_called_once()
        fallback_client.models.generate_content.assert_not_called()

    def test_gemini_maps_roles_and_json_mode_through_google_genai_sdk(self):
        from unittest.mock import Mock
        from google import genai

        fake_client = types.SimpleNamespace(
            models=types.SimpleNamespace(
                generate_content=Mock(return_value=types.SimpleNamespace(text='{"action":"lookup_order"}'))
            )
        )
        with patch.object(genai, "Client", return_value=fake_client) as client_factory:
            model = GeminiChatModel("unit-test-key", "gemini-3.5-flash-lite", timeout=7)
            answer = model.complete([
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Find my order."},
                {"role": "assistant", "content": "Please share the ID."},
                {"role": "user", "content": "12345678"},
            ], json_mode=True)

        self.assertEqual(answer, '{"action":"lookup_order"}')
        client_factory.assert_called_once()
        self.assertEqual(client_factory.call_args.kwargs["api_key"], "unit-test-key")
        self.assertFalse(client_factory.call_args.kwargs["vertexai"])
        self.assertEqual(client_factory.call_args.kwargs["http_options"].retry_options.attempts, 1)
        request = fake_client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.5-flash-lite")
        self.assertEqual(request["config"].system_instruction, "Be concise.")
        self.assertEqual(request["config"].response_mime_type, "application/json")
        self.assertEqual([content.role for content in request["contents"]],
                         ["user", "model", "user"])

    def test_gemini_request_spacing_respects_configured_minimum(self):
        from unittest.mock import Mock
        from google import genai

        fake_client = types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=Mock(side_effect=[
                types.SimpleNamespace(text="first"), types.SimpleNamespace(text="second"),
            ]))
        )
        with patch.object(genai, "Client", return_value=fake_client), \
             patch("voiceops.llm.time.monotonic", side_effect=[100, 100, 101, 105.1]), \
             patch("voiceops.llm.time.sleep") as sleep:
            model = GeminiChatModel("unit-test-key", "gemini-3.5-flash-lite", min_request_interval=4.1)
            model.complete([{"role": "user", "content": "first"}])
            model.complete([{"role": "user", "content": "second"}])

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 3.1)

    def test_gemini_function_call_response_maps_into_existing_tool_action(self):
        from unittest.mock import Mock
        from google import genai

        response = types.SimpleNamespace(
            text=None,
            candidates=[types.SimpleNamespace(
                finish_reason="MALFORMED_FUNCTION_CALL",
                content=types.SimpleNamespace(parts=[types.SimpleNamespace(function_call=
                    types.SimpleNamespace(name="lookup_order", args={
                        "order_id": "00001234", "verification_value": "1234",
                    }))]),
            )],
        )
        fake_client = types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=Mock(return_value=response))
        )
        with patch.object(genai, "Client", return_value=fake_client):
            model = GeminiChatModel("unit-test-key", "gemini-3.5-flash-lite")
            result = model.complete([{"role": "user", "content": "Please check my order."}])
        self.assertEqual(json.loads(result), {
            "action": "lookup_order", "order_id": "00001234", "verification_value": "1234",
        })

    def test_identifiers_are_strings_and_order_requires_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"
            write_csv(data / "customers.csv", ["customer_id", "phone"], [["0007", "9812345678"]])
            write_csv(data / "orders.csv", ["order_id", "customer_id", "delivery_pincode"], [["00042", "0007", "012345"]])
            write_csv(data / "order_items.csv", ["order_id", "sku", "qty", "unit_price_inr"], [["00042", "00A", "1", "99"]])
            write_csv(data / "products.csv", ["sku", "name"], [["00A", "Tea"]])
            write_csv(data / "pincodes.csv", ["pincode"], [["012345"]])
            write_csv(data / "holidays.csv", ["date"], [])
            store = StoreData(root)
            self.assertIn("00042", store.orders)
            self.assertFalse(store.verified_order("00042")["verified"])
            self.assertTrue(store.verified_order("00042", phone_last4="5678")["verified"])
            self.assertTrue(store.verified_order("00042", pincode="012345")["verified"])
            self.assertEqual(store.verified_order("00042", phone_last4="5678")["items"][0]["sku"], "00A")

    def test_business_days_skip_weekends_and_holidays(self):
        self.assertEqual(add_business_days("2026-10-16", 2, {"2026-10-19"}), "2026-10-20")
        self.assertEqual(add_business_days("2026-10-16", 1, set()), "2026-10-17")
        dates = refund_date_context("My refund to net banking was started on 17 October 2026. By which date?", {"2026-10-19"})
        self.assertIn("2026-10-24", dates)
        self.assertIn("2026-10-22 through 2026-10-24", dates)
        overdue = refund_date_context("My UPI refund started 2026-10-11, what now?", {"2026-10-19"})
        self.assertIn("4 business days past due", overdue)
        self.assertIn("more than 3 business days", overdue)

    def test_retrieval_ignores_superseded_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            kb = Path(temp) / "kb"; kb.mkdir()
            (kb / "policy.md").write_text("---\ndoc_id: TEST-1\ndoc_type: policy\nstatus: current\neffective_from: 2026-01-01\n---\n## [RETURN-1] Returns\nReturn within 30 calendar days.", encoding="utf-8")
            (kb / "policy_old.md").write_text("---\ndoc_id: TEST-0\ndoc_type: policy\nstatus: superseded\neffective_from: 2025-01-01\n---\n## [RETURN-0] Returns\nReturn within 90 days.", encoding="utf-8")
            results = KnowledgeBase(Path(temp)).search("return days")
            self.assertTrue(results)
            self.assertEqual(results[0].section_id, "RETURN-1")
            self.assertEqual(results[0].status, "current")
            self.assertTrue(any("30 calendar days" in result.text for result in results))
            # Keep superseded evidence available for the narrow pre-1-August exception.
            self.assertTrue(any("90 days" in result.text for result in results))

    def test_retrieval_practice_evidence_and_hard_negatives(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
        kb = KnowledgeBase(dataset)
        anchors = {chunk.section_id for chunk in kb.chunks
                   if chunk.section_id in {"KB-00.1", "KB-00.2", "KB-00.3"}}
        complete, hard_negative_hits = 0, []
        for row in read_csv(dataset / "practice" / "questions.csv"):
            sources = {chunk.section_id for chunk in kb.search(row["question"])} | anchors
            evidence = [value.strip() for value in row["gold_evidence"].split(" ; ") if value.strip()]
            if evidence and all(value in sources for value in evidence):
                complete += 1
            hard = [value.strip() for value in row["hard_negatives"].split(" ; ") if value.strip()]
            hard_negative_hits.extend((row["id"], value) for value in hard if value in sources)
        self.assertGreaterEqual(complete, 45)
        self.assertEqual(hard_negative_hits, [])

    def test_supplied_dataset_and_mock_tools(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
        if not (dataset / "tools" / "tools.py").exists():
            self.skipTest("Capstone ZIP has not been extracted")
        store = StoreData(dataset)
        tools = AgentTools(store)
        order_id = next(iter(store.orders))
        pending = tools.lookup_order(order_id)
        self.assertTrue(pending["verification_required"])
        self.assertNotIn("status", pending)
        order = store.orders[order_id]
        customer = store.customers[order["customer_id"]]
        verified = tools.lookup_order(order_id, customer["phone"][-4:])
        self.assertTrue(verified["verified"])
        self.assertNotIn("delivery_pincode", verified)
        kb = KnowledgeBase(dataset)
        returns = {chunk.section_id for chunk in kb.search("electronics delivered before 1 august 2026 return 10 days", 8)}
        self.assertIn("RET-3", returns)
        catalog = {chunk.section_id for chunk in kb.search("ELE-0012 warranty", 5)}
        self.assertIn("products.csv:ELE-0012", catalog)
        natural = {chunk.section_id for chunk in kb.search("nobody came to collect my return", 8)}
        self.assertIn("RVP-2", natural)
        offline_kb = KnowledgeBase(dataset, EmbeddingClient("", "https://example.invalid/v1", "text-embedding-3-small"))
        self.assertTrue(offline_kb.search("refund to UPI in how many hours"))
        self.assertIn("OPENAI_API_KEY", offline_kb.embedding_error)
        questions = {row["id"]: row for row in read_csv(dataset / "practice" / "questions.csv")}
        expected_latest = {"PQ0008": "2026-10-27", "PQ0022": "2026-10-27",
                           "PQ0027": "2026-10-15", "PQ0028": "2026-10-21",
                           "PQ0029": "2026-10-21", "PQ0033": "2026-10-24",
                           "PQ0035": "2026-10-17", "PQ0052": "2026-10-24",
                           "PQ0059": "2026-10-22"}
        for question_id, latest in expected_latest.items():
            context = refund_date_context(questions[question_id]["question"], store.holidays)
            self.assertIn(latest, context, question_id)

    def test_agent_verifies_once_keeps_state_and_redacts_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"
            write_csv(data / "customers.csv", ["customer_id", "phone"], [["0007", "9812345678"]])
            write_csv(data / "orders.csv", ["order_id", "customer_id", "delivery_pincode", "status"], [["00042", "0007", "012345", "Shipped"]])
            write_csv(data / "order_items.csv", ["order_id", "sku", "qty", "unit_price_inr"], [])
            write_csv(data / "products.csv", ["sku", "name"], [])
            write_csv(data / "pincodes.csv", ["pincode"], [["012345"]])
            write_csv(data / "holidays.csv", ["date"], [])
            store = StoreData(root)
            tools = AgentTools(store, root / "tickets.jsonl")
            model = MockModel(
                '{"action":"lookup_order","order_id":"00042","verification_value":""}',
                '{"action":"lookup_order","order_id":"00042","verification_value":"5678"}',
                "Your order 00042 is Shipped. Your verification digits were 5678.",
                "Your order is still Shipped.",
            )
            agent = SupportAgent(model, 3, tools=tools)
            first = agent.respond("Where is order 00042?")
            self.assertTrue(first.tool_events[0]["verification_required"])
            self.assertIn("delivery pincode", first.answer)
            self.assertNotIn("shipped", first.answer.casefold())
            # Customers often answer the verification prompt with just the four digits.
            verified = agent.respond("5678")
            self.assertTrue(verified.tool_events[0]["verified"])
            self.assertNotIn("5678", verified.answer)
            self.assertNotIn("5678", str(verified.tool_events))
            next_turn = agent.respond("And now?")
            self.assertEqual(next_turn.tool_events, [])
            self.assertIn("00042", model.prompts[-1][0]["content"])
            self.assertNotIn("5678", str(agent.history))

    def test_bare_order_id_after_prompt_runs_lookup_instead_of_showing_json(self):
        class LookupTools:
            def __init__(self): self.calls = []
            def lookup_order(self, order_id, verification_value=""):
                self.calls.append((order_id, verification_value))
                return {"verified": False, "verification_required": True,
                        "message": "Verification required."}

        model = MockModel(
            "I can help check that. Could you please share your 8-digit order ID?",
            '```json\n{"action":"lookup_order","order_id":"12345667",'
            '"verification_value":""}\nis_voice: false\n```',
        )
        tools = LookupTools()
        agent = SupportAgent(model, 3, tools=tools)

        asked = agent.respond("Where is my order?")
        response = agent.respond("12345667")

        self.assertIn("8-digit order ID", asked.answer)
        self.assertEqual(tools.calls, [("12345667", "")])
        self.assertEqual(response.tool_events, [{
            "action": "lookup_order", "order_id": "12345667",
            "verified": False, "verification_required": True,
        }])
        self.assertIn("last four digits", response.answer)
        self.assertNotIn('"action"', response.answer)
        # The ID-only reply is handled deterministically; Gemini is not called again.
        self.assertEqual(len(model.prompts), 1)

    def test_generic_order_id_prompt_then_sample_pincode_verifies_order(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
        model = MockModel(
            "Hello! To check the status of your order, could you please share your order ID?",
            "Your verified order status is shipped.",
        )
        agent = SupportAgent(model, 3, tools=AgentTools(StoreData(dataset)))

        first = agent.respond("Where is my order?")
        lookup = agent.respond("69520867")
        verified = agent.respond("160017")

        self.assertIn("order ID", first.answer)
        self.assertEqual(lookup.tool_events[0]["action"], "lookup_order")
        self.assertTrue(lookup.tool_events[0]["verification_required"])
        self.assertIn("delivery pincode", lookup.answer)
        self.assertTrue(verified.tool_events[0]["verified"])
        self.assertIn("shipped", verified.answer)
        self.assertNotIn("160017", verified.answer)
        self.assertNotIn("160017", str(agent.history))

    def test_tool_action_parser_accepts_fenced_json_with_trailing_metadata(self):
        text = ('```json\n{"action":"lookup_order","order_id":"12345667",'
                '"verification_value":""}\nis_voice: false\n```')
        self.assertEqual(SupportAgent._parse_tool_action(text), {
            "action": "lookup_order", "order_id": "12345667", "verification_value": "",
        })

    def test_agent_stops_after_two_failed_verification_proofs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"
            write_csv(data / "customers.csv", ["customer_id", "phone"], [["0007", "9812345678"]])
            write_csv(data / "orders.csv", ["order_id", "customer_id", "delivery_pincode", "status"], [["00042", "0007", "012345", "Shipped"]])
            write_csv(data / "order_items.csv", ["order_id", "sku", "qty", "unit_price_inr"], [])
            write_csv(data / "products.csv", ["sku", "name"], [])
            write_csv(data / "pincodes.csv", ["pincode"], [["012345"]])
            write_csv(data / "holidays.csv", ["date"], [])
            model = MockModel(
                '{"action":"lookup_order","order_id":"00042","verification_value":""}',
                '{"action":"lookup_order","order_id":"00042","verification_value":"1111"}',
                '{"action":"lookup_order","order_id":"00042","verification_value":"999999"}',
                '{"action":"lookup_order","order_id":"00042","verification_value":"5678"}',
            )
            agent = SupportAgent(model, 3, tools=AgentTools(StoreData(root), root / "tickets.jsonl"))
            agent.respond("Order 00042 status?")
            agent.respond("Phone last four digits are 1111")
            agent.respond("Delivery pincode is 999999")
            final = agent.respond("Phone last four digits are 5678")
            self.assertFalse(final.tool_events[0]["verified"])
            self.assertFalse(final.tool_events[0]["verification_required"])
            self.assertIn("registered phone", final.answer)
            self.assertNotIn("Shipped", final.answer)
            self.assertEqual(len(model.prompts), 0)

    def test_supplied_mock_ticket_contract(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
        if not (dataset / "tools" / "tools.py").exists():
            self.skipTest("Capstone ZIP has not been extracted")
        tools = AgentTools(StoreData(dataset))
        rejected = tools.create_ticket("tiny")
        self.assertFalse(rejected["created"])
        with tempfile.TemporaryDirectory() as temp:
            tools.mock._TICKETS = str(Path(temp) / "tickets_created.csv")
            created = tools.create_ticket("Verified customer reports a delayed delivery; handover requested.",
                                          order_id="12345678", reason="customer_request", intent="delivery")
            self.assertTrue(created["created"])
            self.assertTrue(Path(tools.mock._TICKETS).exists())

    def test_agent_does_not_claim_ticket_after_tool_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"
            write_csv(data / "customers.csv", ["customer_id", "phone"], [])
            write_csv(data / "orders.csv", ["order_id", "customer_id", "delivery_pincode"], [])
            write_csv(data / "order_items.csv", ["order_id", "line_no", "sku"], [])
            write_csv(data / "products.csv", ["sku", "name"], [])
            write_csv(data / "pincodes.csv", ["pincode"], [])
            write_csv(data / "holidays.csv", ["date"], [])
            model = MockModel(
                '{"action":"create_ticket","summary":"tiny","order_id":"","reason":"","intent":""}',
                "Your ticket has been created.",
            )
            agent = SupportAgent(model, 3, tools=AgentTools(StoreData(root), root / "tickets.jsonl"))
            response = agent.respond("Please escalate this issue.")
            self.assertFalse(response.tool_events[0]["created"])
            self.assertIn("couldn't create", response.answer)
            self.assertNotIn("Your ticket has been created", response.answer)

    def test_pending_failed_proof_retrieves_idv3_guidance(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
        agent = SupportAgent(MockModel(), 3, knowledge=KnowledgeBase(dataset))
        agent.pending_verification_orders.add("77461439")
        _, sources = agent._system("आख़िरी चार अंक 1266 हैं।")
        self.assertIn("IDV-3", sources)

    def test_overdue_verified_refund_forces_confirmed_ticket_and_rrn_guidance(self):
        class TicketTools:
            def __init__(self): self.calls = []
            def create_ticket(self, summary, order_id="", reason="", intent=""):
                self.calls.append((summary, order_id, reason, intent))
                return {"created": True, "ticket_id": "ZM-TEST"}

        tools = TicketTools()
        model = MockModel()
        agent = SupportAgent(model, 3, tools=tools)
        agent.verified_orders["16640894"] = {
            "return_refund_initiated_on": "2026-07-20",
            "return_refund_completed_on": None,
            "return_refund_method": "upi",
            "return_rrn": "315633825001",
        }
        response = agent.respond("Mera refund kahan hai?")
        self.assertEqual([event["action"] for event in response.tool_events], ["create_ticket"])
        self.assertTrue(response.tool_events[0]["created"])
        self.assertIn("315633825001", response.answer)
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(len(model.prompts), 0)

    def test_gnani_adapter_uses_sdk_transcript_and_audio_bytes(self):
        calls = {}

        class FakeSTT:
            def __init__(self): pass
            def transcribe(self, path, language_code):
                calls["stt"] = (path, language_code)
                return {"transcript": "hello"}

        class FakeTTS:
            def __init__(self): pass
            def synthesize(self, text, voice, model, language):
                calls["tts"] = (text, voice, model, language)
                return b"RIFF-fake-wav"

        gnani = types.ModuleType("gnani")
        gnani.__path__ = []
        stt = types.ModuleType("gnani.stt"); stt.GnaniSTTClient = FakeSTT
        tts = types.ModuleType("gnani.tts"); tts.GnaniTTSClient = FakeTTS
        with patch.dict(os.environ, {"GNANI_API_KEY": "unit-test-key"}), patch.dict(sys.modules,
                {"gnani": gnani, "gnani.stt": stt, "gnani.tts": tts}):
            voice = GnaniVoice("unit-test-key")
            self.assertEqual(voice.transcribe(Path("call.wav"), "hi-IN"), "hello")
            target = TEST_TMP / "reply.wav"
            voice.synthesize("**Namaste**\n- Order 12345678\nSources: TRK-1", target, "hi-IN")
        self.assertEqual(calls["stt"], ("call.wav", "hi-IN"))
        self.assertEqual(calls["tts"], ("Namaste Order 1234 5678", "Kaveri", "timbre-v2.5", "hi-IN"))
        self.assertEqual(target.read_bytes(), b"RIFF-fake-wav")


if __name__ == "__main__":
    unittest.main()
