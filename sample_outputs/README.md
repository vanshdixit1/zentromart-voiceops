# Sample outputs

These generated WAVs and JSONL traces are included in the final submission ZIP but ignored by Git to keep the public repository lightweight. The bundled inputs are synthetic practice clips.

`live_voice_smoke_verified/` contains one synthetic clip that completed the live Gnani STT -> Gemini agent -> Gnani TTS path. Its transcript, answer, source IDs, retrieved chunks, and reply audio are saved together. This is not a personal microphone recording; record a clear short phrase yourself before presenting the browser microphone workflow.

To create new outputs locally, set the API keys in `.env`, then run `python batch_run.py <audio-folder> --output sample_outputs`. Use only recordings you own or have permission to process. Do not add `.env`, API keys, or real customer audio to the archive or GitHub.
