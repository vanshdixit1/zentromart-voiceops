from __future__ import annotations

import zipfile
from pathlib import Path

from src.voiceops.config import resolve_dataset

ROOT = Path(__file__).resolve().parent
DATASET = resolve_dataset(ROOT / "dataset" / "Capstone_Project_Dataset_FTTT_2026")
OUTPUT = ROOT / "voiceops_bundle.zip"


def main() -> int:
    if not (DATASET / "README.md").is_file():
        raise SystemExit(
            f"Dataset not found at {DATASET}. Extract the capstone dataset under dataset/ first."
        )
    files = [ROOT / "web_app.py"]
    files.extend(sorted((ROOT / "src").rglob("*.py")))
    files.extend(path for path in sorted((ROOT / "web").rglob("*")) if path.is_file())
    files.extend(
        path for path in sorted(DATASET.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and path.name != "tickets_created.csv"
    )
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print(f"Created {OUTPUT.name}: {len(files)} files, {OUTPUT.stat().st_size:,} bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
