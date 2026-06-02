#!/usr/bin/env python3
"""Print ScaleWoB app split names for shell launchers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "benchmark_apps.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print JAMEL benchmark app split.")
    parser.add_argument("split", choices=("test10", "train86", "all"))
    parser.add_argument("--config", default=str(_default_config()))
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if args.split == "all":
        apps = [*payload["train86"], *payload["test10"]]
    else:
        apps = payload[args.split]
    print(" ".join(apps))


if __name__ == "__main__":
    main()
