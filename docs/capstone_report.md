# VoiceOps 2026 — ZentroMart Capstone Report

## What was built

A four-stage ZentroMart support agent using Google AI Studio through the official `google-genai` SDK (`gemini-3.5-flash-lite`), the supplied documents and mock tools, and Gnani as the only speech provider (`en-IN`, `timbre-v2.5`, Kaveri). The existing OpenAI-compatible embedding endpoint and in-memory vector index remain optional and unchanged; evaluations used offline lexical retrieval. A local Python-standard-library web server offers the same Step 1–3 modes, a Step 3 browser microphone flow, transcript, spoken reply, and retrieved sources while reusing the same Python agent. Audit snapshot: 25 October 2026; dataset reference date: 20 October 2026.

## Evaluation results

Generated-answer evaluation ran against the supplied synthetic practice pack. The full Step 1/2 run used the brief's `asr_text` field for all 52 questions in each step (104 Gemini responses total, 4.1-second configured spacing). The project owner completed manual review of paraphrase-sensitive results; the table reports the evaluator's literal-match scores. Earlier `question`-field results and the diagnostic three-case pilot remain in the evaluation snapshot as historical context.

| Stage | Cases available | Required facts | Evidence | Notes |
| --- | ---: | ---: | ---: | --- |
| Step 1 prompt-only (`asr_text`) | 52/52 | 20.4% (10/49 scored) | 0/47 (prompt-only; no RAG) | 100% passed forbidden-claim and hard-negative checks. |
| Step 2 grounded RAG (`asr_text`) | 52/52 | 59.2% (29/49 scored) | 100% (47/47 scored) | Evidence checks complete; 100% passed forbidden-claim and hard-negative checks. |
| Step 3 single questions | 52/52 | 81.6% (40/49 scored) | 100% (47/47 scored) | Tool-call screening passed 52/52. |
| Step 3 ASR-text legacy snapshot | 52/52 | 71.4% (35/49 scored) | 93.6% (44/47 scored) | Earlier text simulation; no full Step 3 rerun was made for this update. |
| Step 3 conversations (after fixes) | 96/96 | 76.6% (59/77 scored) | 100% (96/96) | Tool-call checks passed 96/96; all 96 evidence checks passed. |
| Step 4 voice | 1 synthetic batch clip plus 1 live browser microphone turn | Both end-to-end checks passed | 11 source IDs saved for the batch clip; sources displayed in the browser | The browser transcribed “Where is my order?”, returned the expected request for an eight-digit order ID, and provided a six-second Gnani reply. The personal recording was not saved. |

### Earlier three-case diagnostic pilot

The first three `asr_text` cases diagnosed spoken-SKU warranty, wallet refund timing, and unsupported midnight delivery. After targeted fixes, their Step 2 recheck scored 3/3 required facts, 2/2 gold-evidence checks, 3/3 hard-negative checks, and 3/3 tool-call checks. This earlier diagnostic sample is separate from the completed full-pack run.

One synthetic audio file completed Gnani transcription, the shared Gemini agent, source capture, and Gnani speech synthesis. Its saved transcript, reply, 11 retrieved chunks and 13.44-second reply WAV are under `sample_outputs/live_voice_smoke_verified/`. The batch runner now measures Gnani streaming WAV files from their actual data length when RIFF uses the streaming-size sentinel.

The local working copy retains the earlier full run and updated conversation-only run under `reports/`; raw reports are excluded from Git and the submission archive. Aggregate results are preserved in `docs/evaluation_snapshot.json`. After reviewing misses, the agent binds lookups to caller-supplied or pending IDs, escalates verified refunds past their maximum timeline, retrieves failed-verification guidance, and confirms a ticket only after `created=true`. Conversation tool-call and evidence checks improved from 76/96 and 45/96 to 96/96 each. The latest evaluation used the configured request spacing and no parallel requests.

## Findings and limits

Retrieval produced complete evidence matches for all 47 scored Step 2 cases. Required-fact exact-match improved from 20.4% in Step 1 to 59.2% in Step 2. The project owner completed manual review of paraphrase-sensitive results; automated fact checks remain screening metrics. All required conversation tool-call and evidence checks pass in the saved post-fix evaluation.

What to improve next: review answer quality with the judging rubric and native speakers if additional language-quality feedback is available. The full Step 1/2 `asr_text` run and project-owner review are complete. For the judging demo, use a quiet room and speak clearly; the deployed browser microphone has completed a live turn. Do not include anyone else's voice without written consent. Bonus work includes the responsive local/hosted web interface, encrypted hosted conversation sessions, and captured retrieval/source outputs.

Offline automated tests: 38 ran; 37 passed and one socket-dependent test was skipped because this environment blocks local socket connections. Built-in Python `trace` measured 81.1% line coverage (1,422/1,754 executable lines) across 10 first-party Python modules exercised by tests; branch coverage was not measured. The dataset validator passed all nine CSVs, 21 knowledge documents, joins, order math, practice evidence, and verification gates. The live voice smoke used one synthetic clip; a separate clear personal browser recording was transcribed and answered with retrieved sources and a six-second spoken reply. That personal recording was not retained. The demo uses synthetic questions; no real customer data or recordings were used.

## Reproducibility

- Repository: https://github.com/vanshdixit1/zentromart-voiceops.
- Runtime: Python 3.14.0; dependencies installed with `python -m pip install -r requirements.txt`.
- Interactive agent: `python run.py --step 3`.
- Local website: `python web_app.py`, then open `http://127.0.0.1:8000`.
- Hosted website: https://zentromart-voiceops.vercel.app.
- Voice batch: `python batch_run.py <audio-folder> --output sample_outputs`.
- Dataset check: `python validate_dataset.py`.
- Evaluation command (uses Gemini quota): `python evaluate.py --dataset dataset --steps 1 2 3`.
- Small quota-conscious sample: `python evaluate.py --dataset dataset --steps 1 2 --limit 3 --output reports/asr-pilot.jsonl`.
- `.env` is local and ignored. Gemini calls are spaced by 4.1 seconds for the 15-requests-per-minute tier. Optional failover uses `GEMINI_FALLBACK_API_KEY` from a different AI Studio project; add the previous key there if desired. Timeouts are not retried automatically. Never publish `.env` or API keys.
