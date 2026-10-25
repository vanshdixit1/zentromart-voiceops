# Two-minute demo recording plan

Record your own voice. The supplied ZIP contains no audio. Do not include real customer data or another person's voice without written consent.

1. **0:00–0:15 — Setup.** Show the running app/terminal and say which LLM, Gnani voice, and language are configured. Keep API keys and `.env` off screen.
2. **0:15–0:45 — Grounded answer.** Ask a new policy question not shown in the reference examples. Show the concise answer and its retrieved section IDs.
3. **0:45–1:20 — Follow-up and tool use.** Ask about a synthetic order. Show the agent asking for one verification value, provide the test customer's own synthetic suffix/pincode, then show the verified answer. Never show the expected proof before the caller supplies it.
4. **1:20–1:45 — Handover or uncertainty.** Ask a policy edge case that requires a human or is not covered. Show the ticket result or the agent's clear uncertainty; do not claim success without a tool confirmation.
5. **1:45–2:00 — Speech reply.** Play the Gnani TTS response and show the saved transcript/audio output.

Use `batch_run.py` to create speech outputs. The video must show actual agent-generated answers; do not type or edit answers into the recording.
