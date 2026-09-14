#!/usr/bin/env python3
"""Held-out eval for IFPruning: dense vs. zero-shot vs. trained vs. static mask.

Measures mean per-token cross-entropy loss (over response tokens only, same
labels-masking convention as training) on a held-out Alpaca slice under four
conditions, to isolate how much of any improvement comes from (a) training
the predictor at all and (b) conditioning the mask on each instruction
specifically rather than using one fixed mask for everything:

  - dense: unpruned target model (upper bound)
  - zero_shot: pruned, per-example dynamic mask from an UNTRAINED predictor
  - trained_dynamic: pruned, per-example dynamic mask from a trained predictor
  - trained_static: pruned, ONE mask (from a fixed generic instruction) via
    the same trained predictor, applied to every example
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json


GENERIC_INSTRUCTION = "Complete the following task to the best of your ability."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    parser.add_argument("--predictor-backbone", default="LiquidAI/LFM2.5-230M-Base")
    parser.add_argument("--mlp-pruning-delta", type=Path, help="Trained delta from mlp_pruning_pilot.py.")
    parser.add_argument("--prune-ratio", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-start-percent", type=int, default=90)
    parser.add_argument("--held-out-end-percent", type=int, default=95)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lqh.experiments.mlp_pruning.model import PruningMaskContext, count_ffn_layers, patch_mlp_with_mask
    from lqh.experiments.mlp_pruning.predictor import SparsityPredictor

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    predictor_tokenizer = AutoTokenizer.from_pretrained(args.predictor_backbone)
    if predictor_tokenizer.pad_token is None:
        predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
    predictor_tokenizer.padding_side = "right"

    held_out = load_dataset(
        "tatsu-lab/alpaca",
        split=f"train[{args.held_out_start_percent}%:{args.held_out_end_percent}%]",
    )
    if args.max_examples:
        held_out = held_out.select(range(min(args.max_examples, len(held_out))))

    def build_prompt(row: dict) -> str:
        if row.get("input"):
            return f"{row['instruction']}\n\n{row['input']}\n\n"
        return f"{row['instruction']}\n\n"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    num_ffn_layers = count_ffn_layers(model)
    ffn_dim = model.config.intermediate_size
    keep_count = max(1, min(ffn_dim - 1, round(ffn_dim * args.prune_ratio)))

    predictor_backbone = AutoModelForCausalLM.from_pretrained(args.predictor_backbone, dtype=torch.bfloat16).to("cuda")
    predictor = SparsityPredictor(predictor_backbone, num_ffn_layers=num_ffn_layers, ffn_dim=ffn_dim)
    predictor.to("cuda")
    if args.mlp_pruning_delta:
        delta = torch.load(args.mlp_pruning_delta, map_location="cpu", weights_only=True)
        predictor.backbone.load_state_dict(delta["predictor_state"]["backbone"])
        predictor.head.load_state_dict(delta["predictor_state"]["head"])
        if delta.get("ffn_state"):
            missing, unexpected = model.load_state_dict(delta["ffn_state"], strict=False)
            if unexpected:
                raise ValueError(f"unexpected ffn_state keys: {unexpected}")
        print(f"loaded trained predictor (+ffn if present) from {args.mlp_pruning_delta}")

    ctx = PruningMaskContext()
    patched = patch_mlp_with_mask(model, ctx)

    def mean_response_loss(mask_provider) -> tuple[float, int]:
        total_nll = 0.0
        total_tokens = 0
        for row in held_out:
            prompt = build_prompt(row)
            full_text = prompt + row["output"] + tokenizer.eos_token
            prompt_ids = predictor_tokenizer(prompt, truncation=True, max_length=args.prompt_length, return_tensors="pt")
            full = tokenizer(full_text, truncation=True, max_length=args.seq_length, return_tensors="pt")
            prompt_len_target_tok = len(
                tokenizer(prompt, truncation=True, max_length=args.prompt_length)["input_ids"]
            )
            labels = full["input_ids"].clone()
            labels[:, :prompt_len_target_tok] = -100
            if (labels != -100).sum().item() == 0:
                continue

            ctx.mask = mask_provider(prompt_ids)
            with torch.no_grad():
                outputs = model(
                    input_ids=full["input_ids"].to("cuda"),
                    attention_mask=full["attention_mask"].to("cuda"),
                    labels=labels.to("cuda"),
                )
            ctx.mask = None
            num_valid = (labels != -100).sum().item()
            total_nll += outputs.loss.item() * num_valid
            total_tokens += num_valid
        return total_nll / total_tokens, total_tokens

    results: dict[str, dict] = {}

    print("evaluating: dense (unpruned)")
    ctx.mask = None
    dense_nll, dense_tokens = mean_response_loss(lambda _prompt_ids: None)
    results["dense"] = {"mean_nll": dense_nll, "token_count": dense_tokens}

    print("evaluating: zero_shot (untrained predictor, dynamic per-example mask)")
    zero_shot_predictor_backbone = AutoModelForCausalLM.from_pretrained(args.predictor_backbone, dtype=torch.bfloat16).to("cuda")
    zero_shot_predictor = SparsityPredictor(zero_shot_predictor_backbone, num_ffn_layers=num_ffn_layers, ffn_dim=ffn_dim)
    zero_shot_predictor.to("cuda")

    def zero_shot_mask(prompt_ids):
        return zero_shot_predictor.predict_mask(
            prompt_ids["input_ids"].to("cuda"), prompt_ids["attention_mask"].to("cuda"), keep_count
        )

    zs_nll, zs_tokens = mean_response_loss(zero_shot_mask)
    results["zero_shot"] = {"mean_nll": zs_nll, "token_count": zs_tokens}

    if args.mlp_pruning_delta:
        print("evaluating: trained_dynamic (trained predictor, dynamic per-example mask)")

        def trained_dynamic_mask(prompt_ids):
            return predictor.predict_mask(
                prompt_ids["input_ids"].to("cuda"), prompt_ids["attention_mask"].to("cuda"), keep_count
            )

        td_nll, td_tokens = mean_response_loss(trained_dynamic_mask)
        results["trained_dynamic"] = {"mean_nll": td_nll, "token_count": td_tokens}

        print("evaluating: trained_static (trained predictor, ONE mask from a generic instruction for all examples)")
        generic_ids = predictor_tokenizer(GENERIC_INSTRUCTION, return_tensors="pt")
        static_mask = predictor.predict_mask(
            generic_ids["input_ids"].to("cuda"), generic_ids["attention_mask"].to("cuda"), keep_count
        )

        def trained_static_mask(prompt_ids):
            batch_size = prompt_ids["input_ids"].shape[0]
            return static_mask.expand(batch_size, -1, -1)

        ts_nll, ts_tokens = mean_response_loss(trained_static_mask)
        results["trained_static"] = {"mean_nll": ts_nll, "token_count": ts_tokens}

    from lqh.experiments.mlp_pruning.model import revert_mlp_patch

    revert_mlp_patch(patched)

    payload = {
        "model": args.model,
        "predictor_backbone": args.predictor_backbone,
        "mlp_pruning_delta": str(args.mlp_pruning_delta) if args.mlp_pruning_delta else None,
        "prune_ratio": args.prune_ratio,
        "keep_count": keep_count,
        "held_out_examples": len(held_out),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
