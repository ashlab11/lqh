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
