from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from voiceops.agent import SupportAgent
from voiceops.config import load_settings, resolve_dataset
from voiceops.data import StoreData
from voiceops.llm import GeminiChatModel, LLMError
from voiceops.retrieval import configured_knowledge
from voiceops.tools import AgentTools


def main() -> None:
    parser = argparse.ArgumentParser(description="ZentroMart text support agent")
    parser.add_argument("--step", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--question", help="Answer one question and exit; omit for a conversation")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    cfg = load_settings()
    dataset = resolve_dataset((args.dataset or cfg.dataset).resolve())
    model = GeminiChatModel(cfg.gemini_api_key, cfg.gemini_model, cfg.gemini_timeout,
                            cfg.gemini_min_request_interval, cfg.gemini_fallback_api_key)
    knowledge = configured_knowledge(dataset, cfg) if args.step >= 2 else None
    tools = AgentTools(StoreData(dataset)) if args.step >= 3 else None
    agent = SupportAgent(model, args.step, knowledge, tools)
    print("ZentroMart AI support assistant. Type 'quit' to end.")
    if args.question:
        questions = [args.question]
    else:
        questions = []
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in {"quit", "exit"}:
                break
            questions.append(q)
            response = agent.respond(q)
            print(f"Agent: {response.answer}")
            if response.sources:
                print("Sources: " + ", ".join(response.sources))
        return
    for question in questions:
        response = agent.respond(question)
        print(response.answer)
        if response.sources:
            print("Sources: " + ", ".join(response.sources))
        if response.tool_events:
            print("Tool events: " + str(response.tool_events))


if __name__ == "__main__":
    try:
        main()
    except (LLMError, FileNotFoundError, ValueError) as exc:
        print(f"Could not run the support agent: {exc}", file=sys.stderr)
        raise SystemExit(2)
