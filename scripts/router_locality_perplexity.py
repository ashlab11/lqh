#!/usr/bin/env python3
"""Held-out Wikitext-2 perplexity for a base or router-delta-patched checkpoint.

Uses the wikitext-2-raw-v1 *test* split (disjoint from the pilot's train[:1%]
fine-tuning slice) with the standard concatenate-then-sliding-window protocol,
so numbers are comparable across baseline/control/treatment checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-Base")
    parser.add_argument("--router-delta", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda").eval()
    if args.router_delta:
        delta = torch.load(args.router_delta, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(delta["router_weights"], strict=False)
        if unexpected:
            raise ValueError(f"unexpected router-delta keys: {unexpected}")
        print(f"loaded router delta from {args.router_delta} ({len(delta['router_weights'])} tensors)")

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row for row in test["text"] if row.strip())
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    sequence_length = input_ids.size(1)

    nll_sum = 0.0
    token_count = 0
    previous_end = 0
    for begin in range(0, sequence_length, args.stride):
        end = min(begin + args.max_length, sequence_length)
        target_len = end - previous_end
        chunk = input_ids[:, begin:end].to("cuda")
        labels = chunk.clone()
        labels[:, :-target_len] = -100
        with torch.inference_mode():
            outputs = model(chunk, labels=labels)
        num_valid = (labels != -100).sum().item()
        nll_sum += outputs.loss.item() * num_valid
        token_count += num_valid
        previous_end = end
        if end == sequence_length:
            break

    mean_nll = nll_sum / token_count
    perplexity = float(torch.exp(torch.tensor(mean_nll)))
    payload = {
        "model": args.model,
        "router_delta": str(args.router_delta) if args.router_delta else None,
        "split": "wikitext-2-raw-v1/test",
        "max_length": args.max_length,
        "stride": args.stride,
        "token_count": token_count,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
