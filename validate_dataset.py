from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset" / "Capstone_Project_Dataset_FTTT_2026"


def find_validator(path: Path) -> Path:
    if (path / "validate_dataset.py").is_file():
        return path / "validate_dataset.py"
    nested = path / "Capstone_Project_Dataset_FTTT_2026" / "validate_dataset.py"
    return nested


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the supplied capstone dataset integrity checks")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Extracted dataset root or its parent folder")
    args = parser.parse_args()
    validator = find_validator(args.dataset.resolve())
    if not validator.is_file():
        print(f"Dataset validator not found under {args.dataset}. Extract the supplied ZIP first.", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(validator)], cwd=validator.parent, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
