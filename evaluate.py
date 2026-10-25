from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from voiceops.agent import SupportAgent
from voiceops.config import load_settings, resolve_dataset
from voiceops.data import StoreData, read_csv
from voiceops.llm import GeminiChatModel, LLMError
from voiceops.retrieval import configured_knowledge
from voiceops.tools import AgentTools


def pick(row: dict[str, str], *names: str) -> str:
    normalized = {k.strip().lower(): v for k, v in row.items()}
    for name in names:
        if normalized.get(name.lower(), "").strip():
            return normalized[name.lower()].strip()
    return ""


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, re.UNICODE))


def score_case(answer: str, sources: list[str], events: list[dict], row: dict[str, str]) -> dict:
    answer_norm = normalize(answer)
    facts = [x.strip() for x in pick(row, "must_include").split(" ; ") if x.strip()]
    required = []
    for fact in facts:
        alternatives = [normalize(x) for x in fact.split(" / ") if normalize(x)]
        required.append(any(x in answer_norm for x in alternatives))
    forbidden = [normalize(x) for x in pick(row, "must_not_include").split(" ; ") if normalize(x)]
    retrieved = set(sources)
    gold = [x.strip() for x in pick(row, "gold_evidence").split(" ; ") if x.strip()]
    hard_negatives = [x.strip() for x in pick(row, "hard_negatives").split(" ; ") if x.strip()]
    expected_calls = pick(row, "expected_tool_calls")
    expected_names = re.findall(r"(lookup_order|create_ticket)\s*\(", expected_calls)
    actual_names = [str(event.get("action", "")) for event in events]
    expected_order = re.search(r"order_id=([0-9]+)", expected_calls)
    order_match = not expected_order or any(event.get("order_id") == expected_order.group(1) for event in events)
    return {
        "must_include_pass": (sum(required) == len(required)) if required else None,
        "must_include_facts_passed": sum(required), "must_include_facts_total": len(required),
        "must_not_include_pass": not any(term in answer_norm for term in forbidden),
        "evidence_complete": all(item in retrieved for item in gold) if gold else None,
        "gold_evidence_found": [item for item in gold if item in retrieved],
        "hard_negatives_avoided": not any(item in retrieved for item in hard_negatives),
        "expected_tool_calls": expected_names,
        "actual_tool_calls": actual_names,
        "tool_calls_match": actual_names == expected_names and order_match,
    }


def summarize(metrics: list[dict]) -> dict:
    result = {"cases": len(metrics), "scoring_note": "String-based checks are screening signals; paraphrases, equivalent date/number forms, and native-language quality still need human review."}
    for key in ("must_include_pass", "must_not_include_pass", "evidence_complete", "hard_negatives_avoided", "tool_calls_match"):
        values = [m[key] for m in metrics if m.get(key) is not None]
        result[key] = {"passed": sum(values), "scored": len(values),
                       "rate": round(sum(values) / len(values), 4) if values else None}
    return result


def report_progress(output_file, count: int) -> None:
    output_file.flush()
    if count % 10 == 0:
        print(f"Evaluated {count} generated cases...", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent-generated practice evaluation")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2])
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.jsonl"))
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing JSONL run, skipping completed cases")
    args = parser.parse_args()
    cfg = load_settings(); dataset = resolve_dataset((args.dataset or cfg.dataset).resolve())
    questions_path = dataset / "practice" / "questions.csv"
    conversations_path = dataset / "practice" / "conversations.csv"
    if not questions_path.exists():
        print(f"Practice questions not found: {questions_path}", file=sys.stderr); return 2
    model = GeminiChatModel(cfg.gemini_api_key, cfg.gemini_model, cfg.gemini_timeout,
                            cfg.gemini_min_request_interval, cfg.gemini_fallback_api_key)
    kb = configured_knowledge(dataset, cfg)
    tools = AgentTools(StoreData(dataset))
    out = args.output; out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    scored: dict[str, list[dict]] = {}
    completed: dict[tuple[str, int, str, str], dict] = {}
    if args.resume and out.exists():
        with out.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Cannot resume: invalid JSON on line {line_number} of {out}: {exc}",
                          file=sys.stderr)
                    return 2
                kind = str(item.get("kind", ""))
                step = int(item.get("step", 0) or 0)
                identity = (str(item.get("practice_id", "")) if kind != "conversation"
                            else str(item.get("conversation_id", "")))
                turn = str(item.get("turn", "")) if kind == "conversation" else ""
                completed[(kind, step, identity, turn)] = item
                bucket = ({"question": f"step_{step}_questions",
                           "asr_text": "step_3_asr_text",
                           "conversation": "step_3_conversations"}.get(kind))
                if bucket:
                    scored.setdefault(bucket, []).append(item)
        if 3 in args.steps and conversations_path.exists():
            conversation_rows = read_csv(conversations_path)
            expected: dict[str, set[tuple[str, int, str, str]]] = {}
            for row in conversation_rows:
                cid = pick(row, "conversation_id", "case_id", "id")
                turn = pick(row, "turn", "turn_number", "sequence")
                if cid and turn:
                    expected.setdefault(cid, set()).add(("conversation", 3, cid, turn))
            incomplete_ids = {cid for cid, keys in expected.items()
                              if not all(key in completed for key in keys)}
            if incomplete_ids:
                # A conversation agent keeps verification and turn history in memory.
                # Replay every turn of an incomplete conversation to rebuild that state
                # accurately; individual turns cannot safely resume in isolation.
                for key in [key for key in completed
                            if key[0] == "conversation" and key[2] in incomplete_ids]:
                    completed.pop(key)
                scored["step_3_conversations"] = [
                    item for item in scored.get("step_3_conversations", [])
                    if item.get("conversation_id") not in incomplete_ids]
                retained = []
                with out.open(encoding="utf-8") as existing:
                    for line in existing:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # malformed rows were rejected above
                        if not (item.get("kind") == "conversation" and
                                item.get("conversation_id") in incomplete_ids):
                            retained.append(line)
                temp_out = out.with_name(out.name + ".resume.tmp")
                temp_out.write_text("".join(retained), encoding="utf-8")
                temp_out.replace(out)
                print(f"Replaying {len(incomplete_ids)} incomplete conversations to restore turn memory.",
                      flush=True)
    conversation_agents: dict[str, SupportAgent] = {}
    with out.open("a" if args.resume else "w", encoding="utf-8") as f:
        for step in args.steps:
            for row in read_csv(questions_path):
                question = pick(row, "asr_text", "question", "customer_text", "prompt")
                if not question:
                    continue
                practice_id = pick(row, "id")
                question_key = ("question", step, practice_id, "")
                item = completed.get(question_key)
                if item is None:
                    agent = SupportAgent(model, step, kb if step >= 2 else None, tools if step >= 3 else None)
                    response = agent.respond(question)
                    item = {"kind": "question", "step": step, "input": question, "answer": response.answer,
                            "sources": response.sources, "retrieved_chunks": response.retrieved_chunks,
                            "tool_events": response.tool_events,
                            "expected_decision": pick(row, "expected_decision", "expected_action", "decision"),
                            "acceptable_decisions": pick(row, "acceptable_decisions"),
                            "answerable": pick(row, "answerable"),
                            "gold_evidence": pick(row, "gold_evidence", "evidence", "expected_evidence"),
                            "hard_negatives": pick(row, "hard_negatives"),
                            "must_include": pick(row, "must_include"),
                            "must_not_include": pick(row, "must_not_include"),
                            "intent": pick(row, "intent"), "practice_id": practice_id}
                    item.update(score_case(response.answer, response.sources, response.tool_events, row))
                    f.write(json.dumps(item, ensure_ascii=False) + "\n"); count += 1
                    report_progress(f, count)
                    completed[question_key] = item
                    scored.setdefault(f"step_{step}_questions", []).append(item)
                # ASR-like text in the pack is a textual Step 4 smoke case, not an audio
                # recording. It exercises the same Step 3 brain without speech API calls.
                asr_text = pick(row, "asr_text")
                asr_key = ("asr_text", 3, practice_id, "")
                if step == 3 and asr_text and asr_text != question and asr_key not in completed:
                    voice_agent = SupportAgent(model, 3, kb, tools)
                    voice_response = voice_agent.respond(asr_text)
                    asr_item = {**item, "kind": "asr_text", "input": voice_agent.redact(asr_text),
                                "answer": voice_response.answer, "sources": voice_response.sources,
                                "retrieved_chunks": voice_response.retrieved_chunks,
                                "tool_events": voice_response.tool_events}
                    asr_item.update(score_case(voice_response.answer, voice_response.sources,
                                               voice_response.tool_events, row))
                    f.write(json.dumps(asr_item, ensure_ascii=False) + "\n")
                    count += 1
                    report_progress(f, count)
                    completed[asr_key] = asr_item
                    scored.setdefault("step_3_asr_text", []).append(asr_item)
        if conversations_path.exists() and 3 in args.steps:
            conversations = read_csv(conversations_path)
            conversations.sort(key=lambda row: (pick(row, "conversation_id", "case_id", "id"),
                                                int(pick(row, "turn", "turn_number", "sequence") or 0)))
            for conversation in conversations:
                cid = pick(conversation, "conversation_id", "case_id", "id")
                turn = pick(conversation, "turn", "turn_number", "sequence")
                conversation_key = ("conversation", 3, cid, turn)
                if conversation_key in completed:
                    continue
                question = pick(conversation, "customer_message", "customer_text", "asr_text", "customer", "utterance", "text")
                if not question:
                    continue
                # Keep one agent and memory per conversation; input order follows CSV order.
                agent = conversation_agents.setdefault(cid, SupportAgent(model, 3, kb, tools))
                response = agent.respond(question)
                case = {"kind": "conversation", "step": 3, "conversation_id": cid,
                                    "turn": turn, "input": agent.redact(question), "answer": response.answer,
                                    "sources": response.sources, "retrieved_chunks": response.retrieved_chunks,
                                    "tool_events": response.tool_events,
                                    "expected_decision": pick(conversation, "expected_decision", "decision"),
                                    "expected_tool_calls": agent.redact(pick(conversation, "expected_tool_calls")),
                                    "gold_evidence": pick(conversation, "gold_evidence", "evidence"),
                                    "must_include": pick(conversation, "must_include"),
                                    "must_not_include": pick(conversation, "must_not_include"),
                                    "scenario": pick(conversation, "scenario")}
                case.update(score_case(response.answer, response.sources, response.tool_events, conversation))
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
                completed[conversation_key] = case
                scored.setdefault("step_3_conversations", []).append(case)
                count += 1
                report_progress(f, count)
    summary_path = out.with_suffix(".summary.json")
    summary = {name: summarize(rows) for name, rows in scored.items()}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {count} agent-generated results to {out}")
    print(f"Evaluation summary: {summary_path}")
    for step in (1, 2):
        metrics = summary.get(f"step_{step}_questions", {})
        evidence = metrics.get("evidence_complete", {})
        grounded = metrics.get("must_include_pass", {})
        print(f"Step {step} (asr_text): grounded {grounded.get('passed', 0)}/{grounded.get('scored', 0)}; "
              f"gold evidence {evidence.get('passed', 0)}/{evidence.get('scored', 0)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LLMError, FileNotFoundError, ValueError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        raise SystemExit(2)
