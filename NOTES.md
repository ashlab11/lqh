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
- Next: rerun the matched pilot (same seed/data/steps) with this second fix
  before drawing any conclusion about whether the locality loss works; all
  prior pilot results (200 and 2000 step, at 1x and 20x weight) used the
  miscalibrated proxy and are not informative about the corrected loss.
