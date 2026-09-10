#!/usr/bin/env python3
"""Small router-only locality-loss pilot for LFM2.5-8B-A1B-Base."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-Base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--union-weight", type=float, default=0.1)
    parser.add_argument("--churn-weight", type=float, default=0.05)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument(
        "--dataset-percent",
        type=int,
        default=1,
        help="Percent of the Wikitext-2 train split to use, e.g. 20 to avoid "
        "many repeated epochs over a tiny slice at high --steps.",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from lqh.experiments.router_locality.loss import (
        LocalityLossConfig,
        RouterLogitCollector,
        enable_router_only_training,
        locality_loss,
    )

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
        batch = tokenizer.pad(rows, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda")
    enabled = enable_router_only_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    collector = RouterLogitCollector()
    handles = collector.attach(model)
    locality = LocalityLossConfig(union_weight=args.union_weight, churn_weight=args.churn_weight, balance_weight=args.balance_weight)

    class PilotTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            attention_mask = inputs["attention_mask"]
            collector.clear()
            outputs = model(**inputs)
            aux, metrics = locality_loss(collector.layer_logits, attention_mask, locality)
            loss = outputs.loss + aux
            self.log({key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()})
            return (loss, outputs) if return_outputs else loss

    train_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = PilotTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=collate)
    try:
        trainer.train()
    finally:
        for handle in handles:
            handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    router_state = {name: value.detach().cpu() for name, value in model.state_dict().items() if name in enabled}
    torch.save({"base_model": args.model, "router_weights": router_state, "steps": args.steps, "union_weight": args.union_weight, "churn_weight": args.churn_weight, "balance_weight": args.balance_weight}, args.output / "router_delta.pt")
    print(f"saved {len(router_state)} router tensors to {args.output / 'router_delta.pt'}")


if __name__ == "__main__":
    main()
