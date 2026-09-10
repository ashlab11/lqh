#!/usr/bin/env python3
"""Compare routing-trace JSON outputs (baseline vs. trained checkpoints)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path, help="trace JSON files, first is treated as baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path, *other_paths = args.traces
    baseline = json.loads(baseline_path.read_text())
    print(f"baseline: {baseline_path.name}")
    print(f"  router_delta={baseline.get('router_delta')}")
    print(f"  {json.dumps(baseline['aggregate'], sort_keys=True)}")
    for path in other_paths:
        payload = json.loads(path.read_text())
        agg = payload["aggregate"]
        print(f"\n{path.name}")
        print(f"  router_delta={payload.get('router_delta')}")
        print(f"  {json.dumps(agg, sort_keys=True)}")
        base_agg = baseline["aggregate"]
        for key in ("mean_unique_experts", "mean_adjacent_jaccard_distance"):
            if key in agg and key in base_agg:
                delta = agg[key] - base_agg[key]
                pct = (delta / base_agg[key] * 100) if base_agg[key] else float("nan")
                print(f"  {key}: {agg[key]:.5f} vs baseline {base_agg[key]:.5f} (delta {delta:+.5f}, {pct:+.2f}%)")


if __name__ == "__main__":
    main()
