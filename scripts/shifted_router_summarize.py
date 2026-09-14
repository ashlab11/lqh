#!/usr/bin/env python3
"""Summarize a shifted-router trace JSON: per-layer prefetch hit/wasted rates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.trace.read_text())
    print(f"model={payload['model']} prompts={payload['prompt_count']}")
    print(f"overall: {json.dumps(payload['aggregate'], sort_keys=True)}")

    by_layer: dict[int, list[dict]] = defaultdict(list)
    for record in payload["records"]:
        by_layer[record["layer"]].append(record)

    print("\nlayer  hit_rate  wasted_rate  real_unique  shifted_unique")
    for layer in sorted(by_layer):
        records = by_layer[layer]
        hit = sum(r["mean_hit_rate"] for r in records) / len(records)
        wasted = sum(r["mean_wasted_rate"] for r in records) / len(records)
        real_unique = sum(r["real_unique_experts"] for r in records) / len(records)
        shifted_unique = sum(r["shifted_unique_experts"] for r in records) / len(records)
        print(f"{layer:5d}  {hit:8.4f}  {wasted:11.4f}  {real_unique:11.2f}  {shifted_unique:14.2f}")


if __name__ == "__main__":
    main()
