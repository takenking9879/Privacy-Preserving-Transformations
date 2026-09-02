#!/usr/bin/env python3
"""Entry point for the privacy-preserving transformation study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.experiment import run_experiments
from src.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Smaller data and config subset")
    args = parser.parse_args()
    run_experiments(quick=args.quick)
    path = generate_report()
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
