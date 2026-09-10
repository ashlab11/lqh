# Notes

## 2026-09-10

- Created branch `experiment/router-locality` from upstream-derived fork commit
  `cd0c683`.
- The first research milestone is observation-only MoE routing traces for the
  LFM2.5-8B-A1B baseline. Do not start GPU training until the trace schema and
  aggregation metrics have been validated.
- Standard LQH SFT/DPO/GRPO paths must remain unchanged. The eventual custom
  trainer/model hooks are opt-in and intended for cluster/local editable-fork
  runs.
- Ran matched 200-step router-only pilots on LFM2.5-8B-A1B-Base (Slurm
  2299974 control / language loss only, 2299975 treatment / union=0.1,
  churn=0.05, balance=0.1) and re-traced both against the untouched baseline
  (2296617). Result: negligible locality shift (mean_unique_experts moved
  -0.11% control, -0.22% treatment vs. baseline; jaccard churn moved <0.15%
  in both). Treatment was not meaningfully different from control.
  See experiments/router_locality/results/compare_baseline_control_treatment.txt.
- Root cause: `locality_loss`'s expected-union term divided an
  already-per-sequence union estimate by valid-token count a second time,
  shrinking its gradient by ~seq_length (256x for these runs) relative to
  the language-model loss, so `union_weight=0.1` was effectively inert.
  Fixed in loss.py (log-space per-sequence expected-union-fraction, no
  extra length division); added regression tests in
  tests/unit/test_router_locality_loss.py. Full pytest still can't run in
  the login env (missing pyarrow in tests/conftest.py); ran the new tests
  standalone.
- Reran the matched pilot with the length-dilution fix at 200 steps (2300455)
  and 2000 steps (2300500, control 2300501): still <0.3% locality shift.
  Tried 20x the weight at 2000 steps (2300575): the union proxy still didn't
  trend downward (oscillated 0.6-0.99) and language-model loss visibly
  worsened (~5.5 vs. ~2.7-3 for control) -- a sign the loss was fighting a
  gradient direction that doesn't correspond to real locality.
- Root cause #2: `locality_loss` renormalized sigmoid(logits) to sum to 1
  across all experts per token, but LFM2's real selection
  (`Lfm2MoeSparseMoeBlock.route_tokens_to_experts` in transformers 5.9's
  modeling_lfm2_moe.py) picks the top_k experts by *raw* sigmoid(logits).
  Renormalizing to sum to 1 regardless of top_k under-counts each token's
  true per-token selection mass (which should sum to top_k=4) by ~4x, and
  makes the loss sensitive to overall logit scale rather than which experts
  are chosen. Fixed by using `top_k * softmax(logits)` (clamped to 1) as the
  per-expert relaxed top-k-membership probability -- since sigmoid and
  softmax are both monotonic in logits, this shares the real mechanism's
  rank order and is correctly calibrated to sum to top_k per token. Added
  tests/unit/test_router_locality_loss.py::test_union_loss_proxy_is_calibrated_to_top_k_not_one.
- Reran the matched 2000-step pilot with the calibration fix (2300759): went
  to NaN loss within ~15 steps. Cancelled the job (`scancel`) rather than
  let it burn GPU time to completion.
- Root cause #3: float32's own precision near 1.0 (~1.19e-7 ULP) is coarser
  than the old `epsilon=1e-8`, so `clamp(max=1-epsilon)` was a silent no-op
  whenever a probability saturated to exactly 1.0 (confirmed directly:
  `torch.tensor(1.0).clamp(max=1-1e-8) == 1.0` in float32). `log1p(-1.0)`'s
  *derivative* is `-inf` (not just the forward value), so real, confidently
  routed logits blew up the backward pass into NaN within ~15 steps.
  Bumped `LocalityLossConfig.epsilon` to 1e-4 (comfortably above float32
  precision; also upcast to float32 before the log1p, since bf16's own ULP
  near 1.0, ~0.0078, is coarser still) and confirmed the clamp now yields a
  clean *zero* gradient past the boundary instead of `-inf`. Added
  tests/unit/test_router_locality_loss.py::test_union_loss_gradient_is_finite_at_probability_saturation
  and ::test_union_loss_does_not_nan_on_bf16_padding (masking before log1p,
  not after, is still correct defensive practice even though it wasn't the
  actual trigger here since the pilot's batch_size=1 runs have no padding).
- Reran the matched 2000-step pilot with all three fixes (2301029): trained
  cleanly (no NaN, train_loss 3.86 vs. control's 3.64). Traced (2301087):
  mean_unique_experts -0.11% vs. baseline -- same negligible magnitude as
  every prior (buggy) attempt. The loss is now numerically correct but the
  real effect is still tiny at union_weight=0.1.
- Scaled 10x (union_weight=1.0, churn_weight=0.5) at 2000 steps, and pulled
  from a 20%-of-train slice (added --dataset-percent /DATASET_PERCENT) to
  avoid repeating a tiny 1% slice for 9 epochs. Control 2301507, treatment
  2301506, traced as 2301655/2301656: treatment moved further than control
  on both metrics (unique_experts -0.22% vs. control's -0.11%; jaccard
  churn -0.50% vs. control's -0.23%) -- a real, consistent, if still small,
  directional signal for the first time.
- Ran held-out Wikitext-2 *test*-split perplexity (scripts/router_locality_perplexity.{py,sbatch},
  disjoint from the pilot's train slice) on baseline/control/both treatment
  weights: baseline 7.119, control 7.049, union_weight=0.1 treatment 7.102
  (within noise of baseline), union_weight=1.0 treatment 7.298 (+2.5%, a
  real quality cost). 10x the weight bought no more locality (-0.22% both
  times) but did cost real perplexity -- a sign of a capacity limit, not a
  tuning problem.
- Diagnosis: `enable_router_only_training` freezes the entire expert bank,
  so it can't specialize around the router's new choices -- but SPEC's
  method 2 description was "SFTing the router and experts" together, not
  router-only. Added `enable_router_and_expert_training` in loss.py (also
  unfreezes each Lfm2MoeExperts module's `gate_up_proj`/`down_proj`
  parameters -- transformers 5.9 stores every expert as two 3D tensors, not
  per-expert nn.Linear submodules) and a `--trainable {router,router_and_experts}`
  flag on scripts/router_locality_pilot.py (`TRAINABLE` in the sbatch
  wrapper). peft/LoRA is not installed in the shared cluster venv, so this
  is full-parameter unfreezing of the expert FFNs, not a LoRA adapter --
  watch GPU memory/optimizer-state size on the first router_and_experts run.
  Added tests/unit/test_router_locality_trainable_scope.py.
- Next: run a matched router_and_experts pilot (same union/churn/balance
  weights as one of the router-only runs above) and compare locality +
  perplexity against the router-only results at the same weight.

## 2026-09-10 (continued): router+experts training

- Implemented `enable_router_and_expert_training` (unfreezes each MoE
  layer's `gate.weight`, `gate_up_proj`, `down_proj`) and a
  `--trainable {router,router_and_experts}` flag on the pilot script, per
  the user's decision to unfreeze experts after router-only training
  proved capacity-limited (see above).
- Smoke test (2302342, 50 steps) succeeded: no OOM/NaN, but delta
  checkpoints are ~15GB (experts are most of the 8B model) -- confirmed
  disk has room (4.3TB free) but this is no longer a "compact" checkpoint.
- Ran matched 2000-step control/treatment (2302363/2302364, same
  union=1.0/churn=0.5/balance=0.1 weights and 20%-of-train data as the
  best router-only run) and traced them (2302656/2302657): **treatment's
  mean_unique_experts dropped 16.37% and jaccard churn dropped 17.78%
  vs. baseline** -- an order of magnitude larger effect than any
  router-only result.
- BUT held-out Wikitext-2 test perplexity collapsed for **both** runs:
  control (no locality loss, LM loss only) 11.11 (+56% vs. baseline's
  7.119), treatment 16.64 (+134%). This means full-parameter fine-tuning
  of the expert bank at lr=1e-4 on ~4400 examples (20% of Wikitext-2
  train) is itself catastrophically destabilizing the model, independent
  of the locality loss -- the large locality "win" is confounded by
  general model collapse, not validated as a quality-preserving
  specialization. Do not report the -16%/-17.78% numbers as a real result
  without first getting the control's perplexity back near baseline.
- Next: fix the destabilization before drawing conclusions -- likely a
  much lower learning rate for the expert parameters specifically (full
  FT of pretrained experts usually wants ~1e-5 or lower, not 1e-4), and/or
  parameter-efficient tuning (LoRA on gate_up_proj/down_proj) instead of
  full-parameter unfreezing. peft is not installed in the shared cluster
  venv; installing it or hand-rolling a low-rank adapter is now on the
  table.

## 2026-09-10 (continued): expert LR fix reveals the -16% result was collapse, not specialization

- Root cause of the router+experts collapse: both control and treatment
  used the router's lr=1e-4 for the expert FFN params too. Full
  fine-tuning of pretrained experts at that LR is itself destabilizing.
  Added --expert-learning-rate (default 1e-5) with separate AdamW param
  groups in PilotTrainer.create_optimizer (router_locality_pilot.py).
- Smoke test at the lower expert LR (2302804, 100 steps/5% data) confirmed
  quality is preserved: perplexity 7.027, matching baseline's 7.119.
- Reran the full matched control/treatment (2302835/2302836, same
  union=1.0/churn=0.5/balance=0.1 weights and 20%-of-train data as the
  collapsed run) with the fixed expert LR. Traced (2303048/2303049) and
  evaluated perplexity (2303050/2303051):
  - control: mean_unique_experts +0.11%, perplexity 6.792 (-4.6%, i.e.
    *better* than baseline)
  - treatment: mean_unique_experts **-0.16%**, jaccard churn -0.33%,
    perplexity 7.088 (-0.4%, essentially matching baseline)
- **Conclusion: the earlier -16.37%/-17.78% locality result (2302364) was
  almost entirely an artifact of the too-high expert LR destabilizing the
  model into a degenerate routing pattern, not real learned
  specialization.** With training actually stable (quality preserved),
  the real, controlled locality effect from router+experts training is
  back down to the same tiny magnitude (~0.1-0.5%) seen in every
  router-only variant tried so far, across three independent
  configurations: union_weight in {0.1, 1.0} router-only, and
  union_weight=1.0 router+experts. This is a fairly strong signal that
  reaching materially more locality (e.g. double-digit percent reductions
  in touched experts) needs either much more training (steps/data), a
  fundamentally different loss design (e.g. penalize churn between
  *adjacent requests*, not just adjacent tokens within one sequence -- the
  premise in SPEC.md is per-request reuse across a whole serving session,
  which this pilot's within-sequence framing only approximates), or
  accepting that post-hoc SFT of a model already trained with a
  load-balancing objective has limited headroom for this without more
  investment.
- Status: infrastructure (trace, loss, both training scopes, perplexity
  eval) is now validated as numerically correct and stable. The locality
  effect achieved so far is real but small. Decision point: continue
  investing in method 2's loss design/scale, or move on to methods 1/3.
