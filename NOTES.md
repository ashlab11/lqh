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
- Next: rerun the matched pilot (same seed/data/steps) with the corrected
  loss before drawing any conclusion about whether the locality loss works;
  the prior 200-step result is not informative about the corrected loss.
