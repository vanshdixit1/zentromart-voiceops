# VoiceOps 2026 — ZentroMart Capstone Report

## What was built

A four-stage ZentroMart support agent using Google AI Studio through the official `google-genai` SDK (`gemini-3.5-flash-lite`), the supplied documents and mock tools, and Gnani as the only speech provider (`en-IN`, `timbre-v2.5`, Kaveri). The existing OpenAI-compatible embedding endpoint and in-memory vector index remain optional and unchanged; evaluations used offline lexical retrieval. A local Python-standard-library web server offers the same Step 1–3 modes, a Step 3 browser microphone flow, transcript, spoken reply, and retrieved sources while reusing the same Python agent. Audit snapshot: 25 October 2026; dataset reference date: 20 October 2026.

## Evaluation results

Generated-answer evaluation ran against the supplied synthetic practice pack. Results are automated string/source checks and need human review for paraphrases, equivalent dates and numbers, and Hindi/Hinglish quality. **The full Step 1/2 table below is a saved historical run using the same `question` field for both steps. A separate, quota-limited live pilot used the brief's `asr_text` field for three cases; it is not a full 52-question rerun.**

| Stage | Cases available | Required facts | Evidence | Notes |
| --- | ---: | ---: | ---: | --- |
| Step 1 prompt-only | 52/52 | 16.3% (8/49 scored) | 0/47 (prompt-only; no RAG) | Baseline; 100% passed forbidden-claim and hard-negative checks. |
| Step 2 grounded RAG | 52/52 | 85.7% (42/49 scored) | 100% (47/47 scored) | Improved grounded coverage over Step 1. |
| Step 3 single questions | 52/52 | 81.6% (40/49 scored) | 100% (47/47 scored) | Tool-call screening passed 52/52. |
| Step 3 ASR-text cases | 52/52 | 71.4% (35/49 scored) | 93.6% (44/47 scored) | Text simulation only; the archive contains no audio. |
| Step 3 conversations (after fixes) | 96/96 | 76.6% (59/77 scored) | 100% (96/96) | Tool-call checks passed 96/96; all 96 evidence checks passed. |
| Step 4 voice | 1 synthetic audio file in this audit | Live end-to-end smoke passed | 11 source IDs saved | Gnani STT → Gemini agent → Gnani TTS completed; browser microphone did not capture intelligible speech in an earlier check. |

### Current quota-limited live pilot

The first three `asr_text` cases were `PQ0002` (spoken SKU warranty), `PQ0003` (wallet refund timing), and `PQ0004` (unsupported midnight delivery). The initial Step 1 baseline scored 1/3 required facts and 0/2 scored evidence checks; initial Step 2 scored 1/3 and 1/2. The Step 2 recheck after the targeted fixes scored 3/3 required facts, 2/2 gold-evidence checks, 3/3 hard-negative checks, and 3/3 tool-call checks. The retrieval fix normalizes spoken SKU digits against actual catalog SKUs; the prompt now says delivery lead-time tables do not establish delivery-slot availability. The unsupported response then abstained rather than inventing delivery hours. These three cases are a smoke sample, not a benchmark result.

One synthetic audio file completed Gnani transcription, the shared Gemini agent, source capture, and Gnani speech synthesis. Its saved transcript, reply, 11 retrieved chunks and 13.44-second reply WAV are under `sample_outputs/live_voice_smoke_verified/`. The batch runner now measures Gnani streaming WAV files from their actual data length when RIFF uses the streaming-size sentinel.

The local working copy retains the earlier full run and updated conversation-only run under `reports/`; raw reports are excluded from Git and the submission archive. Aggregate results are preserved in `docs/evaluation_snapshot.json`. After reviewing misses, the agent binds lookups to caller-supplied or pending IDs, escalates verified refunds past their maximum timeline, retrieves failed-verification guidance, and confirms a ticket only after `created=true`. Conversation tool-call and evidence checks improved from 76/96 and 45/96 to 96/96 each. The latest evaluation used the configured request spacing and no parallel requests.

## Findings and limits

Retrieval produced complete evidence matches for all scored Step 2 cases, and required-fact screening improved from Step 1 to Step 2. The literal fact checks undercount semantically equivalent replies, so their rates are screening signals rather than a final quality judgment. All required conversation tool-call and evidence checks pass in the updated automated evaluation; required-fact checks remain a screening signal and need human review for paraphrases and Hindi/Hinglish quality.

What to improve next: run the full Step 1/2 practice pack on `asr_text` only when quota is available, then review paraphrase-sensitive scores manually. Before a live judging demo, verify microphone positioning and speech level with a clear, short personal recording; do not include anyone else's voice without written consent. Bonus work includes the responsive local/hosted web interface, encrypted hosted conversation sessions, and captured retrieval/source outputs.

Offline automated tests: 38 ran; 37 passed and one socket-dependent test was skipped because this environment blocks local socket connections. Built-in Python `trace` measured 81.1% line coverage (1,422/1,754 executable lines) across 10 first-party Python modules exercised by tests; branch coverage was not measured. The dataset validator passed all nine CSVs, 21 knowledge documents, joins, order math, practice evidence, and verification gates. The live voice smoke used one synthetic clip; the browser microphone was not retested and had returned no intelligible speech in an earlier check. The demo uses synthetic questions; no real customer data or recordings were used.

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
