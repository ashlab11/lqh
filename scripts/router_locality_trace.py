#!/usr/bin/env python3
"""Run an observation-only routing trace for an LFM2 MoE checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


DEFAULT_PROMPTS = [
    "Explain why cache locality matters for high-throughput inference.",
    "Write a short proof that the sum of the first n odd integers is n squared.",
    "Summarize the tradeoff between latency, bandwidth, and memory capacity in a storage hierarchy.",
    "Given a sparse mixture-of-experts language model, describe how routing affects its working-set size.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-Base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--router-delta", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lqh.experiments.router_locality.trace import (
        RouterTraceCollector,
        attach_lfm2_moe_router_trace,
    )

    prompts = json.loads(args.prompts_json.read_text()) if args.prompts_json else DEFAULT_PROMPTS
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts JSON must be a list of strings")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda").eval()
    if args.router_delta:
        delta = torch.load(args.router_delta, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(delta["router_weights"], strict=False)
        if unexpected:
            raise ValueError(f"unexpected router-delta keys: {unexpected}")
        print(f"loaded router delta from {args.router_delta} ({len(delta['router_weights'])} tensors)")

    collector = RouterTraceCollector()
    handles = attach_lfm2_moe_router_trace(model, collector)
    try:
        for prompt_index, prompt in enumerate(prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {name: value.to("cuda") for name, value in encoded.items()}
            before = len(collector.records)
            with torch.inference_mode():
                model(**encoded)
            for record in collector.records[before:]:
                if record.sequence_index != 0:
                    raise RuntimeError("expected one sequence per prompt")
                object.__setattr__(record, "sequence_index", prompt_index)
    finally:
        for handle in handles:
            handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": args.model,
        "router_delta": str(args.router_delta) if args.router_delta else None,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "prompt_count": len(prompts),
        "max_length": args.max_length,
        "aggregate": collector.aggregate(),
        "records": [asdict(record) for record in collector.records],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
