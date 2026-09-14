#!/usr/bin/env python3
"""Held-out Wikitext-2 perplexity under *hard* shifted routing.

Unlike scripts/shifted_router_trace.py (which only measures agreement
between the real and one-layer-early routing decisions), this actually
commits to the shifted decision for every MoE layer's real expert
computation -- no fallback or retrieval for experts the real (post-attention)
state would have picked instead -- and measures the resulting quality cost.

Same protocol as scripts/router_locality_perplexity.py (Wikitext-2 *test*
split, concatenate-then-sliding-window) so numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-Base")
    parser.add_argument(
        "--gate-attr",
        choices=["none", "gate", "shifted_gate"],
        default="none",
        help="'none' evaluates the real, unpatched model. 'gate' commits to "
        "the same (untrained) router fed one layer early. 'shifted_gate' "
        "commits to a router trained by distillation for this purpose "
        "(requires --shifted-gate-delta).",
    )
    parser.add_argument("--shifted-gate-delta", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gate_attr == "shifted_gate" and not args.shifted_gate_delta:
        raise ValueError("--gate-attr=shifted_gate requires --shifted-gate-delta")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lqh.experiments.shifted_router.hard_routing import apply_hard_shifted_routing

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda").eval()

    patched = None
    if args.gate_attr == "shifted_gate":
        from lqh.experiments.shifted_router.model import add_shifted_gates

        add_shifted_gates(model)
        delta = torch.load(args.shifted_gate_delta, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(delta["shifted_gate_weights"], strict=False)
        if unexpected:
            raise ValueError(f"unexpected shifted_gate-delta keys: {unexpected}")
        print(f"loaded shifted_gate delta from {args.shifted_gate_delta} "
              f"({len(delta['shifted_gate_weights'])} tensors)")
    if args.gate_attr != "none":
        patched = apply_hard_shifted_routing(model, gate_attr=args.gate_attr)
        print(f"applied hard shifted routing (gate_attr={args.gate_attr})")

    try:
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
    finally:
        if patched is not None:
            from lqh.experiments.shifted_router.hard_routing import revert_hard_shifted_routing

            revert_hard_shifted_routing(patched)

    mean_nll = nll_sum / token_count
    perplexity = float(torch.exp(torch.tensor(mean_nll)))
    payload = {
        "model": args.model,
        "gate_attr": args.gate_attr,
        "shifted_gate_delta": str(args.shifted_gate_delta) if args.shifted_gate_delta else None,
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
