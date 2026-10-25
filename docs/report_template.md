# VoiceOps 2026 capstone report

Keep the submitted report to four pages maximum. Replace every bracketed field after running the system; do not invent evaluation results.

## 1. What we built

ZentroMart support assistant using [LLM and model], [embedding model or lexical fallback], the supplied local mock tools, and Gnani [STT language/model] plus [TTS voice/model]. Reference date: 20 October 2026.

## 2. Four-step results

| Stage | Questions/cases | Required facts | Evidence coverage | Notes |
| --- | ---: | ---: | ---: | --- |
| Step 1 prompt-only | [from evaluation summary] | [rate] | n/a | [common errors] |
| Step 2 grounded RAG | [from evaluation summary] | [rate] | [rate] | [what improved / failed] |
| Step 3 conversations | 96 turns | [rate] | [rate] | [verification, follow-ups, tools, handovers] |
| Step 4 voice | [audio files you recorded] | [manual review] | [manual review] | [language, clarity, recovery] |

Use `reports/evaluation.summary.json` for screening rates. The evaluator checks literal facts and source IDs; manually inspect semantic paraphrases, dates/numbers in spoken form, and every Hindi/Hinglish response.

## 3. What we learned and fixed

- [Step 1 versus Step 2 comparison]
- [Retrieval miss or hard negative, and the change made]
- [Conversation/tool failure, and the change made]
- [Voice issue, and the change made]

## 4. Limits and next work

[Remaining error patterns, native-speaker review, model limitations, and work you would do next.]

## Reproducibility

- Repository: [public repository URL]
- Python version and install command: [fill in]
- Run command: `python batch_run.py <audio-folder> --output sample_outputs`
- Dataset validator: `python validate_dataset.py`
- Evaluation command: `python evaluate.py --steps 1 2 3`
