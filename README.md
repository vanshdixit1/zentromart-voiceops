# VoiceOps 2026 — ZentroMart Support Agent

A four step capstone implementation for a grounded, conversational support agent. The same agent can answer text, use the supplied order/ticket tools, and handle customer audio through Gnani STT and TTS.

## Setup

1. Extract `Capstone_Project_Dataset_FTTT_2026.zip` into `dataset/`. The expected root is `dataset/Capstone_Project_Dataset_FTTT_2026/` with the provided `README.md`, `DATA_DICTIONARY.md`, `kb/`, `data/`, `practice/`, and `tools/` files. The extracted data folder is ignored by Git.
2. Use Python 3.10 or newer. From the project folder, create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   If PowerShell activation is restricted, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and use `.\.venv\Scripts\python.exe` in place of `python` for the commands below.
3. Copy `.env.example` to `.env`, then set `GEMINI_API_KEY` and `GEMINI_MODEL` for Google AI Studio. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`. Optionally set `GEMINI_FALLBACK_API_KEY` to a key from a different Google AI Studio project. The agent switches keys only after a quota/rate-limit response, tries the primary key again on the next request, and does not retry timeouts automatically. Keys in the same project share quota. The example sets `GEMINI_MIN_REQUEST_INTERVAL_SECONDS=4.1` to stay below a 15-requests-per-minute free-tier limit; lower it only if your project quota allows. The optional `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_EMBEDDING_MODEL` settings are only for the existing embeddings endpoint; without them, retrieval uses its lexical fallback. Step 4 also requires `GNANI_API_KEY`; set `GNANI_STT_LANGUAGE` and `GNANI_TTS_LANGUAGE` separately (defaults: `en-IN`; TTS accepts `hi-en` for Hinglish). Keys are read from the environment and never stored in source.

## Run

```powershell
python validate_dataset.py
python run.py --step 1 --question "Can I cancel an order that has shipped?"
python run.py --step 2 --question "Can I cancel an order that has shipped?"
python run.py --step 3 --question "My order is late."
python evaluate.py --dataset dataset --steps 1 2
python batch_run.py .\customer_audio --output .\sample_outputs
```

## Local website

The website is an optional browser interface over the existing `SupportAgent`, knowledge base, mock tools, and Gnani speech adapter. Its step selector exposes the same Step 1 prompt-only, Step 2 grounded, and Step 3 conversation modes as the PowerShell command. Voice conversations run Step 3 so memory and order tools stay enabled. It runs on Python's standard library server; no separate frontend install is needed. Keep `GEMINI_API_KEY` in `.env` for answers and `GNANI_API_KEY` for microphone input and spoken replies.

In PowerShell, from the project folder, start the website:

```powershell
python web_app.py
```

Open `http://127.0.0.1:8000` in Edge or Chrome and allow microphone access when asked. Select **Record a question**, speak, then select **Stop & send**. The browser sends a short WAV to the existing Gnani STT adapter; the same conversational agent responds, and Gnani returns speech playback and retrieved source IDs. Recordings are limited to 60 seconds. The **Read replies aloud** switch is off by default to avoid extra TTS requests for typed messages; voice messages always receive a spoken reply. Press `Ctrl+C` in the server window to stop it.

The website binds to `127.0.0.1` by default and is for local use. To keep using the PowerShell agent, open a second terminal in the project folder and run `python run.py --step 3`. Steps 1 and 2 remain available through `run.py --step 1` and `run.py --step 2`; the batch submission command above is unchanged. For speech replies, the adapter removes Markdown and source labels and speaks eight-digit order IDs in groups of four.

## Hosted website

Production website: <https://zentromart-voiceops.vercel.app>. The Vercel adapter serves the same `SupportAgent`, dataset, mock tools, and Gnani adapter as the local website. Hosted conversations use an encrypted session token that expires after one hour. Vercel stores the Gemini key, Gnani key, and session-encryption key as sensitive production variables; none belong in this repository. The public URL can consume your Gemini and Gnani quotas, so disable or protect the deployment when you are not using it.

For a fresh deployment, extract the dataset under `dataset/`, run `python build_deploy_bundle.py`, and include the generated `voiceops_bundle.zip` with `api/index.py`, `requirements.txt`, and `vercel.json` in the Vercel deployment. The bundle includes the synthetic dataset but excludes generated ticket logs. The dataset and bundle remain excluded from Git. Configure `GEMINI_API_KEY`, `GEMINI_MODEL`, `GNANI_API_KEY`, `GNANI_STT_LANGUAGE`, `GNANI_TTS_LANGUAGE`, `GNANI_TTS_MODEL`, `GNANI_TTS_VOICE`, and a fresh `VOICEOPS_SESSION_KEY` in Vercel's Production environment before deploying. Generate the session key with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`; enter it only in Vercel's environment-variable settings.

All text generation uses the Google GenAI SDK with the Gemini API through Google AI Studio. Step 1 is prompt-only. Step 2 embeds policy, agent-guide, help, FAQ, catalog, and pincode chunks through the existing configured OpenAI-compatible `/embeddings` endpoint and ranks them in an in-memory cosine vector index. If OpenAI embeddings are unconfigured, unsupported, or unavailable, it falls back to offline lexical retrieval with language-aware query expansion. The embedding model, vector index, and retrieval behavior are unchanged. Current policies take precedence and section IDs stay attached. Step 3 adds conversation memory scoped to one call, the dataset's supplied identity-checked `lookup_order` and `create_ticket` mock functions, and the documented one-retry limit. Step 4 runs each audio file through Gnani STT, the Step 3 agent, then Gnani TTS; it writes a redacted transcript, answer, retrieved source IDs and full retrieved chunk records, tool events, and spoken WAV to the output folder. Clips longer than 60 seconds are rejected and recorded with an error; compressed inputs use Mutagen metadata to check duration. By default each file is a separate call. For multi-turn voice calls, put each caller's turns in its own subfolder and use `--session-mode per-folder`; turns run in filename order. Audio inputs must be the customer's own recordings or otherwise consented recordings.

`evaluate.py` uses the supplied `asr_text` field for Step 1 and Step 2 practice cases, writes generated answers and retrieved chunks, and creates a `.summary.json` with evidence coverage, required-fact checks, forbidden-claim checks, hard-negative retrieval, and expected tool-call comparisons. It prints Step 1 and Step 2 grounded-answer and evidence totals after the run. Its string-based scores are screening signals; review paraphrases, equivalent dates/numbers, and native-language replies manually. The pack contains no audio, so use a permitted personal recording to exercise the live voice path. Public GitHub repository: <https://github.com/vanshdixit1/zentromart-voiceops>.

For long Gemini evaluations, set `GEMINI_MIN_REQUEST_INTERVAL_SECONDS=4.1` when using a 15-requests-per-minute tier. If a run is interrupted, rerun the same command with `--resume`; completed question cases are retained, while incomplete multi-turn conversations are replayed from their first turn so the agent's memory and verification state remain valid.

## Project layout

```text
src/voiceops/       CSV loading, retrieval, agent, tools, and Gnani adapter
web_app.py          local browser server using the existing Step 3 agent
web/                responsive website, microphone capture, and chat UI
run.py              interactive single-turn / conversation entry point
evaluate.py         questions.csv and conversations.csv evaluation runner
batch_run.py        audio-folder batch runner
validate_dataset.py runs the archive's full schema, relational, policy-evidence, and tool-gate validator
tests/              offline tests using small synthetic fixtures
```

See [docs/architecture.md](docs/architecture.md) for the staged workflow and design notes. Data stays local; generated reports and audio should not be committed.

Submission guidance is in [docs/report_template.md](docs/report_template.md) and [docs/demo_script.md](docs/demo_script.md). The report template intentionally has no fabricated scores, and the demo requires a real configured run and your own voice recording.
