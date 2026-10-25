from __future__ import annotations

import json
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from .data import StoreData


class AgentTools:
    def __init__(self, store: StoreData, ticket_log: Path = Path("reports/tickets.jsonl")):
        self.store, self.ticket_log = store, ticket_log
        self.mock = None
        mock_path = store.root / "tools" / "tools.py"
        if mock_path.exists():
            spec = importlib.util.spec_from_file_location("voiceops_dataset_mock_tools", mock_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load supplied mock tools: {mock_path}")
            self.mock = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.mock)

    def lookup_order(self, order_id: str, verification_value: str = "") -> dict:
        if self.mock:
            result = self.mock.lookup_order(order_id, verification_value=verification_value or None)
            # Keep the SDK's complete verified data private to the tool boundary. Only
            # remove identity fields that never need to be sent back to the LLM.
            if result.get("verified"):
                result = {k: v for k, v in result.items()
                          if k not in {"customer_id", "delivery_pincode", "phone", "email", "name"}}
                result.pop("customer_first_name", None)
            return result
        if verification_value:
            digits = "".join(ch for ch in verification_value if ch.isascii() and ch.isdigit())
            result = self.store.verified_order(order_id.strip(), phone_last4=digits, pincode=digits)
            if result and result.get("verified"):
                return result
            return {"verified": False, "verification_required": True,
                    "message": "Verification failed or order not found."}
        return {"verification_required": True, "order_id": order_id,
                "message": "Ask for the registered phone last four digits OR delivery pincode."}

    def create_ticket(self, summary: str, order_id: str = "", reason: str = "", intent: str = "") -> dict:
        if len(summary.strip()) < 20:
            return {"created": False,
                    "message": "Include the issue, checks made, and escalation reason in a short summary."}
        if self.mock:
            return self.mock.create_ticket(summary, order_id=order_id or None,
                                           reason=reason or None, intent=intent or None)
        self.ticket_log.parent.mkdir(parents=True, exist_ok=True)
        ticket_id = f"ZM-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{self.ticket_log.stat().st_size if self.ticket_log.exists() else 0:04d}"
        record = {"ticket_id": ticket_id, "summary": summary.strip(), "created_at": datetime.now(timezone.utc).isoformat()}
        with self.ticket_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"created": True, **record}

