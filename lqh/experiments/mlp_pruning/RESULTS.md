# Method 1 (Apple IFPruning) — results so far

## Objective

Test Apple's Instruction-Following Pruning (Hou et al. 2025,
arxiv.org/abs/2501.02086) on a dense LFM model: prune the MLP intermediate
dimension to 1/4 its size, with a small predictor selecting which channels
to keep per-instruction, rather than a single static pruned subnetwork.

Target: `LiquidAI/LFM2.5-2.6B` (dense, confirmed via its safetensors index —
FFN is 73.5% of its 2.7B params, `intermediate_size=10752`, 30 layers).
Predictor backbone: `LiquidAI/LFM2.5-230M-Base`. Keep ratio: 25%
(10752 → 2688).

## What was built

- `soft_topk.py`: the paper's differentiable top-k mask (eq. 3).
- `model.py`: monkeypatches every `Lfm2MLP` to mask its SwiGLU intermediate
  activation by a per-example, per-layer mask, held fixed for the whole
  sequence.
- `predictor.py`: small LM backbone + 2-layer MLP head, pooling each
  sequence's own last real token.
- `scripts/mlp_pruning_pilot.py`: trains either `predictor_only` (target
  frozen) or `predictor_and_ffn` (also fine-tunes the target's FFN weights,
  matching the paper).
- `scripts/mlp_pruning_eval.py`: held-out Alpaca eval comparing dense,
  zero-shot, trained-dynamic (per-instruction mask), and trained-static
  (one fixed mask for everything) conditions.

Three real-cluster bugs were found and fixed before training worked at all
(dtype mismatches at two boundaries, and a tokenizer-vocab mismatch feeding
the wrong tokenizer's ids into the predictor backbone's embedding table —
see `NOTES.md` for the full bisection trail).

## Results (held-out Alpaca, `train[90%:95%]`, 10 examples)

| condition | predictor_only (1000 steps) | predictor_and_ffn (1000 steps, matched) |
|---|---|---|
| dense (unpruned reference) | 2.07 nll / 8.0 ppl | 2.05 nll / 7.8 ppl |
| zero_shot (untrained predictor) | 14.01 nll / 1,214,884 ppl | 13.31 nll / 602,000 ppl |
| trained_dynamic (per-instruction mask) | **10.58 nll / 39,263 ppl** | 11.85 nll / 139,000 ppl |
| trained_static (one fixed mask) | **10.54 nll / 37,810 ppl** | 11.75 nll / 127,000 ppl |

## Conclusion

1. **The predictor learns something real.** 1000 steps of `predictor_only`
   training took held-out perplexity from ~1.2M (untrained) down to ~38-39K
   — a large, reproducible improvement. But this is still catastrophically
   far from the dense model's 8.0: freezing the target model caps how good
   a 25%-width subnetwork can be, no matter how well the predictor picks
   channels (the same capacity-ceiling pattern seen in method 2's
   router-only training).
2. **Per-instruction conditioning isn't earning its keep yet.**
   `trained_dynamic` and `trained_static` (one mask for every example) are
   nearly identical in both configurations. At this training budget, the
   predictor hasn't learned to meaningfully differentiate its mask by
   instruction — it's converged to roughly one "good enough" subnetwork.
3. **Unfreezing the FFN made things *worse*, not better, at matched
   training budget** (11.75-11.85 vs. 10.54-10.58 nll) — the opposite of
   what the paper's own joint-optimization approach would predict. Training
   showed severe gradient instability (raw grad norm up to 127,000, vs.
   near-zero for the frozen-target run) even at a conservative
   `ffn_learning_rate=1e-5`. The FFN-tuned model's own *unpruned* quality
   stayed intact (~2.05 nll, matching the untouched baseline), so the
   instability is localized to the pruned-forward-pass gradient, not a
   general model breakdown.

## What this session did *not* replicate from the paper

- **Continued pretraining stage.** The paper trains in two stages
  (continued pretraining on chunked raw text, then SFT) specifically
  because "learning to select input-specific sub-networks may require a
  lot of training data." This session only ran a single SFT-only stage on
  a small Alpaca slice.
- **Training scale.** The paper trains for 60K SFT steps with batch size
  1024 on millions of examples; this session's pilots used at most 1000
  steps at batch size 2.
- **Gradient-instability root cause.** Not yet diagnosed why the FFN
  gradient is so much sharper/spikier than the router-weight case in
  method 2 (which needed only a 10x lower LR, not clean stability at
  1e-5 vs. this run's visible blow-ups) — worth investigating before
  concluding predictor_and_ffn doesn't work at all.

## Status

Paused pending direction: diagnose/fix the FFN gradient instability, scale
up predictor_only training further to see where its ceiling actually is, or
attempt the paper's continued-pretraining stage before drawing a final
conclusion about this method's viability at this scale.
