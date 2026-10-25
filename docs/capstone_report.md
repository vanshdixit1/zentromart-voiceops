# VoiceOps 2026 — ZentroMart Capstone Report

## What was built

A four-stage ZentroMart support agent using Google AI Studio through the official `google-genai` SDK (`gemini-3.5-flash-lite`), the supplied documents and mock tools, and Gnani as the only speech provider (`en-IN`, `timbre-v2.5`, Kaveri). The existing OpenAI-compatible embedding endpoint and in-memory vector index remain optional and unchanged; evaluations used offline lexical retrieval. A local Python-standard-library web server offers the same Step 1–3 modes, a Step 3 browser microphone flow, transcript, spoken reply, and retrieved sources while reusing the same Python agent. Audit snapshot: 25 October 2026; dataset reference date: 20 October 2026.

## Evaluation results

Generated-answer evaluation ran against the supplied synthetic practice pack. Results are automated string/source checks and need human review for paraphrases, equivalent dates and numbers, and Hindi/Hinglish quality. **The Step 1/2 table below is a saved historical run using the same `question` field for both steps; the current evaluator defaults to the brief's `asr_text` field, so this is not a fresh `asr_text` run. It was not rerun in this final audit to conserve Gemini quota.**

| Stage | Cases available | Required facts | Evidence | Notes |
| --- | ---: | ---: | ---: | --- |
| Step 1 prompt-only | 52/52 | 16.3% (8/49 scored) | 0/47 (prompt-only; no RAG) | Baseline; 100% passed forbidden-claim and hard-negative checks. |
| Step 2 grounded RAG | 52/52 | 85.7% (42/49 scored) | 100% (47/47 scored) | Improved grounded coverage over Step 1. |
| Step 3 single questions | 52/52 | 81.6% (40/49 scored) | 100% (47/47 scored) | Tool-call screening passed 52/52. |
| Step 3 ASR-text cases | 52/52 | 71.4% (35/49 scored) | 93.6% (44/47 scored) | Text simulation only; the archive contains no audio. |
| Step 3 conversations (after fixes) | 96/96 | 76.6% (59/77 scored) | 100% (96/96) | Tool-call checks passed 96/96; all 96 evidence checks passed. |
| Step 4 voice | 3 synthetic turns + unclear-audio case | Earlier live smoke passed | Sources saved | Gnani STT/TTS and silent-audio retry verified in an earlier run; browser microphone did not capture intelligible speech in a later check. |

The local working copy retains the earlier full run and updated conversation-only run under `reports/`; raw reports are excluded from Git and the submission archive. Aggregate results are preserved in `docs/evaluation_snapshot.json`. After reviewing misses, the agent binds lookups to caller-supplied or pending IDs, escalates verified refunds past their maximum timeline, retrieves failed-verification guidance, and confirms a ticket only after `created=true`. Conversation tool-call and evidence checks improved from 76/96 and 45/96 to 96/96 each. The latest evaluation used the configured request spacing and no parallel requests.

## Findings and limits

Retrieval produced complete evidence matches for all scored Step 2 cases, and required-fact screening improved from Step 1 to Step 2. The literal fact checks undercount semantically equivalent replies, so their rates are screening signals rather than a final quality judgment. All required conversation tool-call and evidence checks pass in the updated automated evaluation; required-fact checks remain a screening signal and need human review for paraphrases and Hindi/Hinglish quality.

What to improve next: rerun Steps 1 and 2 against the brief's `asr_text` field when API quota is available, then review paraphrase-sensitive scores manually. Before a live judging demo, verify microphone positioning and speech level with a clear, short personal recording; do not include anyone else's voice without written consent. Bonus work includes the responsive local/hosted web interface, encrypted hosted conversation sessions, and captured retrieval/source outputs.

Offline automated tests: 36 ran; 35 passed and one socket-dependent test was skipped because this environment blocks local socket connections. The dataset validator passes all nine CSVs, 21 knowledge documents, joins, order math, practice evidence, and verification gates. A live three-turn Gnani audio batch and separate silent-audio recovery case passed in an earlier run; neither was repeated in this audit. The demo uses synthetic questions; no real customer data or recordings were used. The physical microphone did not produce an intelligible transcript in a browser check; Gnani returned its no-speech recovery response. Speech quality should be checked with a clear personal recording before the live judging demo.

## Reproducibility

- Repository: https://github.com/vanshdixit1/zentromart-voiceops.
- Runtime: Python 3.14.0; dependencies installed with `python -m pip install -r requirements.txt`.
- Interactive agent: `python run.py --step 3`.
- Local website: `python web_app.py`, then open `http://127.0.0.1:8000`.
- Hosted website: https://zentromart-voiceops.vercel.app.
- Voice batch: `python batch_run.py <audio-folder> --output sample_outputs`.
- Dataset check: `python validate_dataset.py`.
- Evaluation command (uses Gemini quota): `python evaluate.py --dataset dataset --steps 1 2 3`.
- `.env` is local and ignored. Gemini calls are spaced by 4.1 seconds for the 15-requests-per-minute tier. Optional failover uses `GEMINI_FALLBACK_API_KEY` from a different AI Studio project; add the previous key there if desired. Timeouts are not retried automatically. Never publish `.env` or API keys.
