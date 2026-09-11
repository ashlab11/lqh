#!/usr/bin/env python3
"""Instruction-following MLP pruning pilot for LiquidAI/LFM2.5-2.6B (dense).

First pilot: freeze the target model entirely and train only the sparsity
predictor (a small LM backbone + head) to select, per instruction, which
FFN intermediate-dimension channels of the (frozen) target model to keep.
This tests whether a good per-instruction subset even exists in an
off-the-shelf model before also fine-tuning the target model itself
(--trainable predictor_and_ffn), which is what Hou et al. 2025 do but is a
bigger, riskier step (see this repo's method-2 notes on full-parameter
fine-tuning instability).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    parser.add_argument("--predictor-backbone", default="LiquidAI/LFM2.5-230M-Base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prune-ratio", type=float, default=0.25, help="Fraction of FFN dim to KEEP.")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--trainable",
        choices=["predictor_only", "predictor_and_ffn"],
        default="predictor_only",
        help="'predictor_only' freezes the target model entirely (safer first "
        "pilot: tests whether a good per-instruction FFN subset exists at "
        "all). 'predictor_and_ffn' also fine-tunes the target model's FFN "
        "weights, matching Hou et al.'s joint optimization -- use a low "
        "--ffn-learning-rate given this repo's experience with full "
        "fine-tuning instability (see lqh/experiments/router_locality/RESULTS.md).",
    )
    parser.add_argument("--ffn-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--dataset-percent",
        type=int,
        default=5,
        help="Percent of tatsu-lab/alpaca to use.",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from lqh.experiments.mlp_pruning.model import PruningMaskContext, count_ffn_layers, patch_mlp_with_mask
    from lqh.experiments.mlp_pruning.predictor import SparsityPredictor

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # The predictor backbone is a *different* model (LFM2.5-230M-Base has a
    # 65536-token vocab vs. the target's 128000), so its input must be
    # tokenized with its own tokenizer -- feeding it target-model token ids
    # crashed a real cluster run as an out-of-bounds embedding lookup (an
    # unchecked GPU memory access, not a catchable Python exception).
    predictor_tokenizer = AutoTokenizer.from_pretrained(args.predictor_backbone)
    if predictor_tokenizer.pad_token is None:
        predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
    predictor_tokenizer.padding_side = "right"

    raw = load_dataset("tatsu-lab/alpaca", split=f"train[:{args.dataset_percent}%]")

    def build_prompt(row: dict) -> str:
        if row.get("input"):
            return f"{row['instruction']}\n\n{row['input']}\n\n"
        return f"{row['instruction']}\n\n"

    def tokenize(row: dict) -> dict:
        prompt = build_prompt(row)
        full_text = prompt + row["output"] + tokenizer.eos_token
        prompt_ids = predictor_tokenizer(prompt, truncation=True, max_length=args.prompt_length)["input_ids"]
        full = tokenizer(full_text, truncation=True, max_length=args.seq_length)
        labels = list(full["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        return {"input_ids": full["input_ids"], "attention_mask": full["attention_mask"], "labels": labels, "prompt_ids": prompt_ids}

    dataset = raw.map(tokenize, remove_columns=raw.column_names)
    dataset = dataset.filter(lambda row: any(label != -100 for label in row["labels"]))

    def collate(rows: list[dict]) -> dict:
        full_batch = tokenizer.pad(
            [{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows],
            return_tensors="pt",
        )
        max_label_len = full_batch["input_ids"].shape[1]
        labels = torch.full((len(rows), max_label_len), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            labels[i, : len(row["labels"])] = torch.tensor(row["labels"])
        full_batch["labels"] = labels
        prompt_batch = predictor_tokenizer.pad([{"input_ids": r["prompt_ids"]} for r in rows], return_tensors="pt")
        full_batch["prompt_input_ids"] = prompt_batch["input_ids"]
        full_batch["prompt_attention_mask"] = prompt_batch["attention_mask"]
        return full_batch

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad = False
    ffn_parameters: list[str] = []
    if args.trainable == "predictor_and_ffn":
        for name, parameter in model.named_parameters():
            if ".feed_forward." in name:
                parameter.requires_grad = True
                ffn_parameters.append(name)
    print(f"target model trainable: {args.trainable} ({len(ffn_parameters)} FFN tensors unfrozen)")

    num_ffn_layers = count_ffn_layers(model)
    ffn_dim = model.config.intermediate_size
    keep_count = max(1, min(ffn_dim - 1, round(ffn_dim * args.prune_ratio)))
    print(f"pruning {ffn_dim} -> {keep_count} FFN channels ({args.prune_ratio:.0%}) across {num_ffn_layers} layers")

    predictor_backbone = AutoModelForCausalLM.from_pretrained(args.predictor_backbone, dtype=torch.bfloat16).to("cuda")
    predictor = SparsityPredictor(predictor_backbone, num_ffn_layers=num_ffn_layers, ffn_dim=ffn_dim)
    predictor.to("cuda")

    mask_context = PruningMaskContext()
    patched = patch_mlp_with_mask(model, mask_context)

    class PredictorTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                param_groups = [{"params": list(predictor.parameters()), "lr": args.learning_rate}]
                if args.trainable == "predictor_and_ffn":
                    named = dict(model.named_parameters())
                    param_groups.append({"params": [named[n] for n in ffn_parameters], "lr": args.ffn_learning_rate})
                self.optimizer = torch.optim.AdamW(param_groups)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            mask_context.mask = predictor.predict_mask(
                inputs["prompt_input_ids"], inputs["prompt_attention_mask"], keep_count
            )
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            mask_context.mask = None
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    train_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = PredictorTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=collate)
    try:
        trainer.train()
    finally:
        from lqh.experiments.mlp_pruning.model import revert_mlp_patch

        revert_mlp_patch(patched)

    args.output.mkdir(parents=True, exist_ok=True)
    predictor_state = {
        "backbone": predictor.backbone.state_dict(),
        "head": predictor.head.state_dict(),
    }
    ffn_state = (
        {name: value.detach().cpu() for name, value in model.state_dict().items() if name in ffn_parameters}
        if ffn_parameters
        else {}
    )
    torch.save(
        {
            "target_model": args.model,
            "predictor_backbone": args.predictor_backbone,
            "num_ffn_layers": num_ffn_layers,
            "ffn_dim": ffn_dim,
            "keep_count": keep_count,
            "prune_ratio": args.prune_ratio,
            "trainable": args.trainable,
            "predictor_state": predictor_state,
            "ffn_state": ffn_state,
        },
        args.output / "mlp_pruning_delta.pt",
    )
    print(f"saved predictor ({'+ ffn' if ffn_state else ''}) delta to {args.output / 'mlp_pruning_delta.pt'}")


if __name__ == "__main__":
    main()
