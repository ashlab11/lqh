#!/usr/bin/env python3
"""Distill each MoE layer's router into a shifted_gate for LFM2.5-8B-A1B-Base.

Pure self-distillation: no labels, no language-model loss, and the real
model (including the real router) is completely untouched -- only the new
shifted_gate parameters (one small nn.Linear per MoE layer) are trained.
See lqh/experiments/shifted_router/{model,distill}.py for the design.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-Base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--dataset-percent",
        type=int,
        default=5,
        help="Percent of the Wikitext-2 train split to use.",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from lqh.experiments.shifted_router.distill import ShiftedGateDistillCollector, shifted_gate_distill_loss
    from lqh.experiments.shifted_router.model import add_shifted_gates

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=f"train[:{args.dataset_percent}%]")
    raw = raw.filter(lambda row: bool(row["text"].strip()))

    def tokenize(row):
        return tokenizer(row["text"], truncation=True, max_length=args.seq_length)

    dataset = raw.map(tokenize, remove_columns=raw.column_names)
    dataset = dataset.filter(lambda row: len(row["input_ids"]) > 8)

    def collate(rows):
        # No labels needed: this is pure self-distillation on router logits,
        # not language-model training.
        return tokenizer.pad(rows, return_tensors="pt")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda")
    enabled = add_shifted_gates(model)
    print(f"trainable: shifted_gate ({len(enabled)} tensors, "
          f"{sum(model.get_parameter(n).numel() for n in enabled):,} params)")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    collector = ShiftedGateDistillCollector()
    handles = collector.attach(model)

    class DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            collector.clear()
            outputs = model(**inputs)
            loss, metrics = shifted_gate_distill_loss(collector.layer_captures)
            self.log({key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()})
            return (loss, outputs) if return_outputs else loss

    train_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = DistillTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=collate)
    try:
        trainer.train()
    finally:
        for handle in handles:
            handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    shifted_gate_state = {name: value.detach().cpu() for name, value in model.state_dict().items() if name in enabled}
    torch.save(
        {
            "base_model": args.model,
            "shifted_gate_weights": shifted_gate_state,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
        },
        args.output / "shifted_gate_delta.pt",
    )
    print(f"saved {len(shifted_gate_state)} shifted_gate tensors to {args.output / 'shifted_gate_delta.pt'}")


if __name__ == "__main__":
    main()
