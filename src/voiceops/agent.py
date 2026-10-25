from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import GeminiChatModel
from .retrieval import KnowledgeBase
from .tools import AgentTools
from .data import read_csv, refund_date_context

BASE = """You are ZentroMart's AI support assistant. Introduce yourself on the first reply. Reply in the customer's language and keep written/voice replies to two or three short sentences. Use 20 October 2026 as today's date for this dataset. Never invent policy, dates, amounts, order facts, or completed actions. Follow policy precedence: current policy > agent guide > help article > FAQ > marketing. A superseded policy applies only to cases its successor explicitly preserves; check the order date when relevant. Historical support logs are not authority. If the documents do not answer, say so and offer a human colleague. Refuse compensation or exceptions policy does not allow. If a human decision is required, create a ticket with the issue, intent, order ID, what was told/checked, and escalation reason; never include a verification code, phone suffix, or pincode in its summary. Confirm a ticket only when the tool returns created=true. Never reveal order details before successful tool verification. Ask for one verification value at a time (phone suffix or delivery pincode), never read out the expected value, and allow one retry after an incorrect value. After two failed proofs, stop and offer general policy help/support from the registered phone. Keep verified order facts only within this conversation. Never repeat verification secrets in replies, tickets, or logs. For voice, write short natural spoken replies, include the year in dates, speak order IDs in groups of four, read amounts in rupees, and avoid Markdown, bullets, and source labels."""


@dataclass
class AgentResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    tool_events: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict[str, str]] = field(default_factory=list)


class SupportAgent:
    def __init__(self, model: GeminiChatModel, step: int, knowledge: KnowledgeBase | None = None,
                 tools: AgentTools | None = None):
        if step not in (1, 2, 3):
            raise ValueError("Text agent step must be 1, 2, or 3.")
        self.model, self.step, self.knowledge, self.tools = model, step, knowledge, tools
        self.history: list[dict[str, str]] = []
        self.verified_orders: dict[str, dict] = {}
        self.verification_failures: dict[str, int] = {}
        self.verification_secrets: set[str] = set()
        self.pending_verification_orders: set[str] = set()
        self.escalated_refund_orders: set[str] = set()
        self.retrieved_chunks: list[dict[str, str]] = []
        self.holidays: set[str] = set()
        if self.knowledge:
            holiday_path = self.knowledge.dataset / "data" / "holidays.csv"
            if holiday_path.exists():
                self.holidays = {row.get("date", "") for row in read_csv(holiday_path)}

    def _system(self, question: str) -> tuple[str, list[str]]:
        system = BASE
        sources: list[str] = []
        self.retrieved_chunks = []
        if self.step >= 2:
            recent_context = " ".join(message["content"] for message in self.history[-4:] if message["role"] == "user")
            retrieval_query = f"{recent_context} {question}".strip()
            date_context = refund_date_context(retrieval_query, self.holidays)
            overdue_refund = self._overdue_refund()
            if overdue_refund and re.search(r"refund|रिफंड|रिफ़ंड", retrieval_query, re.I):
                date_context = "\n\n".join(part for part in (date_context, overdue_refund[1]) if part)
            if date_context and "maximum date is" in date_context:
                retrieval_query += " refund escalation handover more than 3 business days late"
            chunks = self.knowledge.search(retrieval_query, k=8) if self.knowledge else []
            # These sections govern precedence, the frozen date, style, and safe
            # verification/escalation behavior; keep them available every turn.
            anchor_ids = {"KB-00.1", "KB-00.2", "KB-00.3", "STY-1"}
            verification_context = bool(self.pending_verification_orders) or bool(
                re.search(r"(?<!\d)(?:[0-9]{8}|[0-9]{4}[\s-][0-9]{4})(?!\d)", retrieval_query)) or bool(
                re.search(r"phone|pincode|pin\b|last\s*(?:four|4)|verification", retrieval_query, re.I))
            if verification_context:
                anchor_ids.add("IDV-2")
            if (any(count > 0 for count in self.verification_failures.values()) or
                    (self.pending_verification_orders and self._proof_values(question))):
                anchor_ids.add("IDV-3")
            if overdue_refund and re.search(r"refund|रिफंड|रिफ़ंड", retrieval_query, re.I):
                anchor_ids.update({"ESC-1", "ESC-2", "ESC-3", "ESC-4", "REF-2", "REF-6"})
            if self.verified_orders and any(order.get("is_plus") for order in self.verified_orders.values()):
                anchor_ids.add("PLS-2")
            if (self.verified_orders and re.search(r"refund|रिफंड|रिफ़ंड", retrieval_query, re.I)
                    and re.search(r"return|pickup|वापस|वापसी|रिटर्न", retrieval_query, re.I)):
                anchor_ids.add("RET-7")
            if any(section.startswith("ESC-") for section in (chunk.section_id for chunk in chunks)):
                anchor_ids.update({"ESC-1", "ESC-2", "ESC-3", "ESC-4"})
            if self.knowledge:
                anchors = [chunk for chunk in self.knowledge.chunks if chunk.section_id in anchor_ids]
                known = {chunk.section_id for chunk in chunks}
                chunks.extend(chunk for chunk in anchors if chunk.section_id not in known)
            sources = list(dict.fromkeys(c.section_id for c in chunks))
            self.retrieved_chunks = [
                {"section_id": c.section_id, "source": c.source, "doc_type": c.doc_type,
                 "status": c.status, "effective_from": c.effective_from, "text": c.text}
                for c in chunks
            ]
            context = "\n\n".join(f"[{c.section_id}; {c.doc_type}; status={c.status}; effective_from={c.effective_from}]\n{c.text}" for c in chunks)
            system += "\n\nUse only the retrieved documents/catalog rows below as factual evidence. A retrieved passage must explicitly support the requested fact; keyword overlap or adjacent subject matter is not enough. For example, delivery lead-time tables do not establish delivery-slot availability or operating hours. Do not infer an unstated option or policy. Apply the precedence stated above. Do not treat a superseded section as generally current; use it only if a current successor explicitly preserves it for this case. If evidence is missing, say you do not know. Keep the section IDs for evaluation.\n\n" + (context or "No relevant knowledge or catalog records were retrieved.")
            if date_context:
                system += "\n\n" + date_context
        return system, sources

    def _overdue_refund(self) -> tuple[str, str, str] | None:
        """Return an overdue, uncompleted return-refund ticket context, if any."""
        for order_id, result in self.verified_orders.items():
            order = result.get("order", result)
            initiated = str(order.get("return_refund_initiated_on") or "").strip()
            completed = str(order.get("return_refund_completed_on") or "").strip()
            method = str(order.get("return_refund_method") or order.get("payment_method", "")).strip()
            if not initiated or completed:
                continue
            calendar = refund_date_context(f"{method} refund initiated {initiated}", self.holidays)
            match = re.search(r"maximum date is (\d+) business days past due", calendar)
            if match and int(match.group(1)) > 3:
                rrn = str(order.get("return_rrn", "") or order.get("return_refund_rrn", ""))
                return order_id, calendar, rrn
        return None

    def redact(self, text: str) -> str:
        for secret in self.verification_secrets:
            if secret:
                text = text.replace(secret, "[verification withheld]")
        return text

    @staticmethod
    def _tool_failure_reply(question: str, action: str, *, retry: bool = False,
                            request_proof: bool = False,
                            missing_order_id: bool = False) -> str:
        hindi = bool(re.search(r"[\u0900-\u097f]", question))
        if action == "create_ticket":
            return ("मैं अभी सहायता टिकट नहीं बना पाया, इसलिए टिकट बनने की पुष्टि नहीं कर सकता। कृपया दोबारा कोशिश करें या किसी सहायता प्रतिनिधि से संपर्क करें।"
                    if hindi else
                    "I couldn't create a support ticket, so I can't confirm one was created. Please try again or contact a human colleague.")
        if missing_order_id:
            return "कृपया पहले 8 अंकों का ऑर्डर आईडी साझा करें।" if hindi else "Please share the 8-digit order ID first."
        if request_proof:
            return ("ऑर्डर की जानकारी सुरक्षित रखने के लिए, रजिस्टर्ड फोन के आख़िरी चार अंक या डिलीवरी पिनकोड साझा करें।"
                    if hindi else
                    "To protect your order details, please share either the last four digits of the registered phone or the delivery pincode.")
        if retry:
            return ("मैं इस जानकारी से पहचान की पुष्टि नहीं कर पाया। कृपया फोन के आख़िरी चार अंक या डिलीवरी पिनकोड जाँचकर एक बार और कोशिश करें।"
                    if hindi else
                    "I couldn't verify that value. Please check the phone's last four digits or the delivery pincode and try once more.")
        return ("पहचान की पुष्टि नहीं हो सकी, इसलिए मैं ऑर्डर का विवरण साझा नहीं कर सकता। कृपया रजिस्टर्ड फोन से सहायता लें।"
                if hindi else
                "I couldn't verify the order, so I can't share its details. Please contact support from the registered phone.")

    def _capture_verification_proof(self, question: str) -> None:
        verification_context = bool(self.pending_verification_orders)
        if not verification_context and not re.search(
                r"phone|pincode|pin\b|last\s*(?:four|4)|digits|number ends", question, re.I):
            return
        for proof in re.findall(r"(?<![0-9])(?:[0-9]{4}|[0-9]{6})(?![0-9])", question):
            self.verification_secrets.add(proof)

    @staticmethod
    def _order_ids(text: str) -> list[str]:
        # Dataset IDs are eight digits. Also accept shorter explicit IDs so small
        # synthetic fixtures and any documented legacy IDs retain their flow.
        if not re.search(r"order|ऑर्डर", text, re.I):
            return []
        return [re.sub(r"\D", "", match) for match in re.findall(
            r"(?<![0-9])(?:[0-9]{5,8}|[0-9]{4}[\s-][0-9]{4})(?![0-9])", text)]

    def _bare_order_id_reply(self, question: str) -> str | None:
        """Recognize an ID-only reply when the prior assistant explicitly asked for one."""
        if not self.history or self.history[-1].get("role") != "assistant":
            return None
        previous = self.history[-1].get("content", "")
        asked_for_id = re.search(
            r"(?:8[- ]digit|eight[- ]digit).{0,30}(?:order\s*)?(?:id|number)|"
            r"(?:share|provide|send|tell).{0,40}(?:your\s+)?(?:8[- ]digit\s+)?order\s*(?:id|number)|"
            r"(?:order\s*(?:id|number)).{0,40}(?:share|provide|send|tell)",
            previous, re.I,
        )
        if not asked_for_id:
            return None
        match = re.fullmatch(r"\s*(\d{4}[\s-]\d{4}|\d{8})\s*[.!]?\s*", question)
        return re.sub(r"\D", "", match.group(1)) if match else None

    @staticmethod
    def _parse_tool_action(text: str) -> dict | None:
        """Read a tool action even if Gemini wraps its JSON with fence/metadata text."""
        candidate_text = text.strip()
        if candidate_text.startswith("```"):
            candidate_text = re.sub(r"^```[^\r\n]*\r?\n?", "", candidate_text, count=1)
        start = candidate_text.find("{")
        if start < 0 or candidate_text[:start].strip():
            return None
        try:
            candidate, _ = json.JSONDecoder().raw_decode(candidate_text, start)
        except json.JSONDecodeError:
            return None
        if isinstance(candidate, dict) and candidate.get("action") in {"lookup_order", "create_ticket"}:
            return candidate
        return None

    @staticmethod
    def _proof_values(text: str) -> list[str]:
        return re.findall(r"(?<![0-9])(?:[0-9]{4}|[0-9]{6})(?![0-9])", text)

    @staticmethod
    def _explicit_human_request(text: str) -> bool:
        return bool(re.search(
            r"\b(speak|talk|connect|transfer)\b.{0,35}\b(human|person|representative|agent)\b|"
            r"\b(human|person|representative|agent)\b.{0,35}\b(speak|talk|connect)\b|"
            r"kisi insaan|insaan se baat|kisi aadmi|human agent|"
            r"इंसान से बात|किसी इंसान|मानव से बात|व्यक्ति से बात|आदमी से बात", text, re.I))

    @staticmethod
    def _handover_commitment(text: str) -> bool:
        if re.search(r"cannot|can't|couldn't|unable to|not able to", text, re.I):
            return False
        return bool(re.search(
            r"priority resolution team|human colleague|human agent will|will connect you|"
            r"will hand (?:this|it) over|ticket.{0,30}(?:create|raise|open)|"
            r"(?:create|raise|open).{0,30}ticket|ticket.{0,20}(?:created|raised|opened)|"
            r"(?:ticket|handover).{0,20}(?:bana|banaya|create|escalat)", text, re.I))

    def _ticket_summary(self, question: str, reason: str) -> str:
        context = [message["content"] for message in self.history[-8:] if message["role"] == "user"]
        context.append(question)
        issue = self.redact(" | ".join(context))
        return (f"Customer issue and conversation: {issue}. Intent: human handover. "
                f"Checks made: reviewed the available policy and conversation context. Escalation reason: {reason}.")

    @staticmethod
    def _ticket_success_reply(question: str, overdue_refund: tuple[str, str, str] | None = None) -> str:
        if overdue_refund:
            rrn = overdue_refund[2]
            if re.search(r"[\u0900-\u097f]", question) or re.search(r"kahan|mera refund", question, re.I):
                trace = f" बैंक इसे reference number {rrn} से trace कर सकता है।" if rrn else ""
                return f"आपका रिफ़ंड तय समय से देर से है। मैंने हमारी टीम से जाँच के लिए टिकट बनाया है।{trace}"
            trace = f" Your bank can trace it using reference number {rrn}." if rrn else ""
            return f"Your refund is past its expected window, so I created a ticket for follow-up.{trace}"
        if re.search(r"[\u0900-\u097f]", question) or re.search(r"kisi insaan|aadmi|refund kahan", question, re.I):
            return "मैंने आपकी सहायता के लिए टिकट बना दिया है। हमारी टीम आपसे संपर्क करेगी।"
        return "I’ve created a support ticket for a human colleague to review. They’ll follow up; I can’t promise a specific outcome or time."

    def respond(self, question: str) -> AgentResponse:
        self._capture_verification_proof(question)
        system, sources = self._system(question)
        events: list[dict] = []
        if self.step < 3:
            answer = self.model.complete([{"role": "system", "content": system},
                                          {"role": "user", "content": question}])
            return AgentResponse(answer.strip(), sources, retrieved_chunks=self.retrieved_chunks.copy())

        cached_orders = json.dumps(self.verified_orders, ensure_ascii=False)
        action_rules = """For an order-specific request, use the supplied lookup_order tool. Return one JSON object only for an action:
{"action":"lookup_order","order_id":"8 digits","verification_value":""}
or {"action":"create_ticket","summary":"...","order_id":"","reason":"...","intent":"..."}.
For the first lookup, call lookup_order with an empty verification_value; the tool will return only a prompt. Ask for exactly one proof value. When the customer provides a suffix or pincode, pass only that value in verification_value. Reuse the order ID from the recent conversation if the customer replies with only the proof. Never guess a proof or read out the expected value. After a failed proof, ask once for a different/checked value; after the second failed proof do not call lookup again and do not share details. For verified order facts available in conversation state, do not ask for verification again. Do not include proof values in tool summaries, tickets, or customer-facing replies. Return ordinary concise customer text when no action is needed."""
        messages = [{"role": "system", "content": system + "\n\n" + action_rules + "\n\nVerified order state for this conversation only: " + cached_orders}]
        messages.extend(self.history[-10:])
        messages.append({"role": "user", "content": question})
        order_ids = self._order_ids(question)
        bare_order_id = self._bare_order_id_reply(question)
        pending_ids = list(self.pending_verification_orders)
        proof_values = self._proof_values(question)
        recent_user_context = " ".join(message["content"] for message in self.history[-8:]
                                       if message["role"] == "user")
        refund_context = bool(re.search(r"refund|रिफंड|रिफ़ंड", f"{recent_user_context} {question}", re.I))
        overdue_refund = self._overdue_refund() if refund_context else None
        forced_action = None
        if self._explicit_human_request(question):
            known_order_ids = list(self.verified_orders)
            forced_action = {"action": "create_ticket", "summary": self._ticket_summary(
                question, "customer explicitly requested a human agent"),
                "order_id": known_order_ids[-1] if len(known_order_ids) == 1 else "",
                "reason": "customer_request", "intent": "human_handover"}
        elif overdue_refund and overdue_refund[0] not in self.escalated_refund_orders:
            forced_action = {"action": "create_ticket", "summary": self._ticket_summary(
                question, "refund is more than three business days past its maximum timeline"),
                "order_id": overdue_refund[0], "reason": "refund_overdue", "intent": "refund_followup"}
        elif len(order_ids) == 1 and order_ids[0] not in self.verified_orders:
            # An 8-digit order ID is unambiguous in this dataset. Verify before
            # generation so the model cannot omit lookup or hallucinate another ID.
            forced_action = {"action": "lookup_order", "order_id": order_ids[0],
                             "verification_value": ""}
        elif bare_order_id and bare_order_id not in self.verified_orders:
            # Customers commonly answer an explicit ID request with only the digits.
            # Bind that reply to the lookup instead of asking Gemini to infer a tool call.
            forced_action = {"action": "lookup_order", "order_id": bare_order_id,
                             "verification_value": ""}
        elif len(pending_ids) == 1 and proof_values:
            # Bind a caller-provided proof to the one pending order, never to an ID
            # guessed by the model from redacted conversation history.
            forced_action = {"action": "lookup_order", "order_id": pending_ids[0],
                             "verification_value": proof_values[-1]}
        first = (json.dumps(forced_action, ensure_ascii=False) if forced_action
                 else self.model.complete(messages).strip())
        action = self._parse_tool_action(first)
        safe_tool_failure_reply = None
        if forced_action:
            action = forced_action
        elif (("ESC-1" in sources and self._handover_commitment(first)) or
              bool(re.search(r"ticket.{0,24}(?:created|raised|opened)|(?:created|raised|opened).{0,24}ticket", first, re.I))):
            # The model identified a mandatory escalation but skipped the tool.
            # Create the required handover before telling the customer it is done.
            forced_action = {"action": "create_ticket", "summary": self._ticket_summary(
                question, "policy requires human review"),
                "order_id": list(self.verified_orders)[-1] if len(self.verified_orders) == 1 else "",
                "reason": "policy_escalation", "intent": "human_handover"}
            action = forced_action
        if action and action.get("action") == "lookup_order" and not forced_action:
            # Never execute a model-invented order ID. Bind to the caller's ID,
            # the sole pending verification, or the already verified order.
            if len(order_ids) == 1:
                action["order_id"] = order_ids[0]
            elif len(pending_ids) == 1 and proof_values:
                action = {"action": "lookup_order", "order_id": pending_ids[0],
                          "verification_value": proof_values[-1]}
            elif len(self.verified_orders) == 1:
                # Ask Gemini to answer from the verified state, with tool use
                # explicitly disabled for this turn rather than calling a fake ID.
                guarded = messages + [{"role": "system", "content":
                    "This conversation already has a verified order. Do not call tools or introduce another order ID. "
                    "Answer the current question from the verified order state and retrieved documents."}]
                first = self.model.complete(guarded).strip()
                action = None
            else:
                first = self._tool_failure_reply(question, "lookup_order", missing_order_id=True)
                action = None
        deterministic_ticket = bool(forced_action and forced_action.get("action") == "create_ticket")
        if action:
            if not self.tools:
                result = {"error": "Tools are unavailable; do not claim the action succeeded."}
                event = {"action": action["action"], "succeeded": False}
                safe_tool_failure_reply = self._tool_failure_reply(question, action["action"])
            elif action["action"] == "lookup_order":
                oid = "".join(ch for ch in str(action.get("order_id", "")) if ch.isascii() and ch.isdigit())
                proof = "".join(ch for ch in str(action.get("verification_value", "")) if ch.isascii() and ch.isdigit())
                supplied_groups = re.findall(r"(?<!\d)(?:\d{4}|\d{6})(?!\d)", question)
                # A prior proof in model context is not authorization to reuse it.
                if proof and proof not in supplied_groups:
                    proof = ""
                if proof and oid:
                    self.verification_secrets.add(proof)
                if oid in self.verified_orders:
                    result = self.verified_orders[oid]
                elif self.verification_failures.get(oid, 0) >= 2:
                    result = {"verified": False, "verification_required": False,
                              "message": "Maximum verification attempts reached. Do not disclose order details."}
                else:
                    result = self.tools.lookup_order(oid, proof)
                    if result.get("verified"):
                        self.verified_orders[oid] = result
                        self.pending_verification_orders.discard(oid)
                    elif result.get("verification_required"):
                        self.pending_verification_orders.add(oid)
                        if proof:
                            self.verification_failures[oid] = self.verification_failures.get(oid, 0) + 1
                    else:
                        self.pending_verification_orders.discard(oid)
                if not result.get("verified"):
                    safe_tool_failure_reply = self._tool_failure_reply(
                        question, "lookup_order",
                        retry=bool(oid and proof and self.verification_failures.get(oid, 0) < 2),
                        request_proof=bool(oid and not proof and self.verification_failures.get(oid, 0) == 0),
                        missing_order_id=not oid,
                    )
                event = {"action": "lookup_order", "order_id": oid,
                         "verified": bool(result.get("verified")),
                         "verification_required": bool(result.get("verification_required"))}
            else:
                summary = str(action.get("summary", ""))
                for secret in self.verification_secrets:
                    summary = summary.replace(secret, "[verification withheld]")
                result = self.tools.create_ticket(summary, str(action.get("order_id", "")),
                                                  str(action.get("reason", "")), str(action.get("intent", "")))
                if not result.get("created"):
                    safe_tool_failure_reply = self._tool_failure_reply(question, "create_ticket")
                event = {"action": "create_ticket", "created": bool(result.get("created")),
                         "ticket_id": result.get("ticket_id", ""),
                         "order_id": str(action.get("order_id", ""))}
                if result.get("created") and action.get("reason") == "refund_overdue":
                    self.escalated_refund_orders.add(str(action.get("order_id", "")))
            events.append(event)
            safe_action = dict(action)
            safe_action.pop("verification_value", None)
            messages.extend([{"role": "assistant", "content": json.dumps(safe_action, ensure_ascii=False)},
                             {"role": "user", "content": "Tool result (authoritative): " + json.dumps(result, ensure_ascii=False) + "\nNow reply to the customer. Do not claim verification or ticket creation unless the result confirms it."}])
            if safe_tool_failure_reply:
                answer = safe_tool_failure_reply
            elif deterministic_ticket:
                answer = self._ticket_success_reply(question, overdue_refund)
            else:
                answer = self.model.complete(messages).strip()
        else:
            answer = first
        # Keep proofs out of conversation history used for later prompts and outputs.
        safe_question = self.redact(question)
        answer = self.redact(answer)
        self.history.extend([{"role": "user", "content": safe_question}, {"role": "assistant", "content": answer}])
        return AgentResponse(answer, sources, events, self.retrieved_chunks.copy())

