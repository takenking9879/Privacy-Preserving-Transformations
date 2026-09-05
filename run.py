#!/usr/bin/env python3
"""Run the utility / privacy transformation protocol."""

from __future__ import annotations

import argparse

from src.experiment import run_experiments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smoke protocol")
    args = p.parse_args()
    df = run_experiments(quick=args.quick)
    print(df.head() if df is not None else "no rows")


if __name__ == "__main__":
    main()
