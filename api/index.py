"""Vercel serverless adapter for the existing VoiceOps web service."""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

_runtime_root = Path(tempfile.gettempdir()) / "voiceops-runtime"
_bundle = Path(__file__).resolve().parents[1] / "voiceops_bundle.zip"
if not (_runtime_root / "web_app.py").is_file():
    _runtime_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_bundle) as _archive:
        _archive.extractall(_runtime_root)

os.environ.setdefault(
    "VOICEOPS_DATASET",
    str(_runtime_root / "dataset" / "Capstone_Project_Dataset_FTTT_2026"),
)
if not os.environ.get("VOICEOPS_SESSION_KEY"):
    raise RuntimeError("VOICEOPS_SESSION_KEY must be configured for secure hosted conversations.")
sys.path.insert(0, str(_runtime_root / "src"))
sys.path.insert(0, str(_runtime_root))

from web_app import VoiceOpsWebService, make_handler  # noqa: E402

_service = VoiceOpsWebService()


class handler(make_handler(_service)):
    pass
