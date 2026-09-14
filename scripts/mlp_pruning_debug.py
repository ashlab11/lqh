#!/usr/bin/env python3
"""Bisect the HSA hardware-exception crash in the mlp_pruning pilot.

Runs a sequence of increasingly complex forward passes, printing a
checkpoint (flushed) after each succeeds, so the last printed line before a
crash pinpoints the failing stage even though the crash itself carries no
Python traceback.
"""

from __future__ import annotations

import sys


def checkpoint(msg: str) -> None:
    print(f"[CHECKPOINT] {msg}", flush=True)
    sys.stdout.flush()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "LiquidAI/LFM2.5-2.6B"
    predictor_name = "LiquidAI/LFM2.5-230M-Base"

    checkpoint("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    checkpoint("loading predictor tokenizer (different vocab from target model!)")
    predictor_tokenizer = AutoTokenizer.from_pretrained(predictor_name)
    if predictor_tokenizer.pad_token is None:
        predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
    predictor_tokenizer.padding_side = "right"

    checkpoint("loading target model")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
    model.eval()
    checkpoint("target model loaded")

    encoded = tokenizer(["Hello, how are you?", "This is another sentence."], return_tensors="pt", padding=True)
    encoded = {k: v.to("cuda") for k, v in encoded.items()}

    checkpoint("stage 1: unpatched target model forward")
    with torch.no_grad():
        out = model(**encoded)
    checkpoint(f"stage 1 OK, logits shape {tuple(out.logits.shape)}")

    from lqh.experiments.mlp_pruning.model import PruningMaskContext, count_ffn_layers, patch_mlp_with_mask

    num_ffn_layers = count_ffn_layers(model)
    ffn_dim = model.config.intermediate_size
    checkpoint(f"num_ffn_layers={num_ffn_layers} ffn_dim={ffn_dim}")

    ctx = PruningMaskContext()
    patched = patch_mlp_with_mask(model, ctx)
    checkpoint("stage 2: patched forward with mask=None (should be passthrough)")
    with torch.no_grad():
        out2 = model(**encoded)
    checkpoint(f"stage 2 OK, matches unpatched: {torch.equal(out.logits, out2.logits)}")

    batch = encoded["input_ids"].shape[0]
    checkpoint("stage 3: patched forward with all-ones mask (no actual pruning)")
    ctx.mask = torch.ones(batch, num_ffn_layers, ffn_dim, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        out3 = model(**encoded)
    checkpoint(f"stage 3 OK, matches unpatched: {torch.allclose(out.logits, out3.logits, atol=1e-2)}")

    keep_count = round(ffn_dim * 0.25)
    checkpoint(f"stage 4: patched forward with real sparse mask (keep_count={keep_count})")
    mask = torch.zeros(batch, num_ffn_layers, ffn_dim, dtype=torch.bfloat16, device="cuda")
    mask[:, :, :keep_count] = 1.0
    ctx.mask = mask
    with torch.no_grad():
        out4 = model(**encoded)
    checkpoint(f"stage 4 OK, logits shape {tuple(out4.logits.shape)}")

    from lqh.experiments.mlp_pruning.model import revert_mlp_patch

    revert_mlp_patch(patched)
    checkpoint("stage 5: loading predictor backbone")
    predictor_backbone = AutoModelForCausalLM.from_pretrained(predictor_name, dtype=torch.bfloat16).to("cuda")
    checkpoint("stage 5 OK, predictor backbone loaded")

    predictor_encoded = predictor_tokenizer(
        ["Hello, how are you?", "This is another sentence."], return_tensors="pt", padding=True
    )
    predictor_encoded = {k: v.to("cuda") for k, v in predictor_encoded.items()}

    checkpoint("stage 6: predictor backbone forward (using its OWN tokenizer's ids)")
    with torch.no_grad():
        predictor_out = predictor_backbone(**predictor_encoded, output_hidden_states=True)
    checkpoint(f"stage 6 OK, hidden_states[-1] shape {tuple(predictor_out.hidden_states[-1].shape)}")

    from lqh.experiments.mlp_pruning.predictor import SparsityPredictor

    checkpoint("stage 7: SparsityPredictor.predict_mask (softmax/topk on real backbone output)")
    predictor = SparsityPredictor(predictor_backbone, num_ffn_layers=num_ffn_layers, ffn_dim=ffn_dim)
    predictor.to("cuda")
    predicted_mask = predictor.predict_mask(predictor_encoded["input_ids"], predictor_encoded["attention_mask"], keep_count)
    checkpoint(f"stage 7 OK, predicted mask shape {tuple(predicted_mask.shape)}")

    checkpoint("stage 8: full pipeline -- patched target forward with predicted mask")
    patched = patch_mlp_with_mask(model, ctx)
    ctx.mask = predicted_mask
    with torch.no_grad():
        out8 = model(**encoded)
    checkpoint(f"stage 8 OK, logits shape {tuple(out8.logits.shape)}")
    revert_mlp_patch(patched)

    checkpoint("stage 9: confirm the predictor actually receives gradients from the LM loss")
    patched = patch_mlp_with_mask(model, ctx)
    labels = encoded["input_ids"].clone()
    ctx.mask = predictor.predict_mask(predictor_encoded["input_ids"], predictor_encoded["attention_mask"], keep_count)
    outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], labels=labels)
    outputs.loss.backward()
    ctx.mask = None
    revert_mlp_patch(patched)

    head_first_layer = predictor.head[0]
    grad_norm = head_first_layer.weight.grad.norm().item() if head_first_layer.weight.grad is not None else None
    checkpoint(f"stage 9: predictor.head[0].weight.grad norm = {grad_norm}")
    backbone_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.backbone.parameters())
    checkpoint(f"stage 9: any predictor.backbone parameter has nonzero grad = {backbone_has_grad}")
    if grad_norm is None or grad_norm == 0:
        raise RuntimeError("predictor received NO gradient from the LM loss -- graph is disconnected somewhere")

    checkpoint("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
