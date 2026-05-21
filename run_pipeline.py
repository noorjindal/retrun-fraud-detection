#!/usr/bin/env python3
"""
run_pipeline.py — Run all ML layers in order.

    python run_pipeline.py              # full pipeline
    python run_pipeline.py --skip-data  # reuse existing raw_returns.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: Path, label: str) -> None:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset generation")
    args = parser.parse_args()

    steps = []
    if not args.skip_data:
        steps.append((ROOT / "data" / "generate_dataset.py", "Layer 1 — Dataset"))
    steps += [
        (ROOT / "features" / "engineer_features.py", "Layer 2 — Feature Engineering"),
        (ROOT / "model" / "train.py", "Layer 3 — Model Training"),
        (ROOT / "model" / "evaluate.py", "Layer 4 — Evaluation"),
    ]

    for script, label in steps:
        if not script.exists():
            print(f"Missing: {script}")
            sys.exit(1)
        run(script, label)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  API:       python backend/app.py")
    print("  Dashboard: streamlit run dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
