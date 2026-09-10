# Method 2 (router-locality SFT) — results so far

## Objective

Fine-tune LFM2.5-8B-A1B-Base so that tokens in a request reuse a small,
stable set of MoE experts, without materially degrading quality (see
`SPEC.md`). If it worked, this would let an SSD-backed expert cache serve
most of a request from a small resident set instead of touching ~65% of
experts per layer per request (the baseline headroom this project
established first).

## Baseline

Untouched LFM2.5-8B-A1B-Base, traced over a fixed 4-prompt suite (22 MoE
layers, 88 sequence/layer records):

| metric | value |
|---|---|
| mean unique experts per sequence/layer (of 32) | 20.76 |
| mean adjacent-token Jaccard churn | 0.692 |
| held-out Wikitext-2 test perplexity | 7.119 |

## What was tried

A differentiable locality loss (`lqh/experiments/router_locality/loss.py`)
combining a per-sequence expected-expert-union term, an adjacent-token
churn term, and a batch-level load-balance term, added to the language
model loss during router-only or router+expert SFT on a Wikitext-2 slice.

Three implementation bugs were found and fixed in sequence (a length-
dilution bug that made the union term's gradient ~256x too weak, a
selection-mechanism mismatch that miscalibrated it ~4x, and a numerical
bug that produced `-inf` gradients at probability saturation — see
`NOTES.md` for the full diagnostic trail). Once the loss was numerically
correct and stable, it was tested at two training scopes:

1. **Router-only** (freeze everything but the gate weights) — cheap, but
   capacity-limited: 10x the loss weight over 2000 steps produced no
   larger locality shift, only real perplexity cost.
2. **Router + experts** (also unfreeze each MoE layer's expert FFN
   weights) — matches the original spec's "SFT the router and experts."
   The expert parameters need a much lower learning rate than the
   router's (1e-5 vs. 1e-4); using the router's LR for both destabilized
   the model into an apparent "-16% locality" result that evaporated once
   training was actually stable (see below).

## Final, validated results (stable training, matched control/treatment)

All runs: 2000 steps, 20% of Wikitext-2 train (to stay under 1 epoch),
`union_weight=1.0, churn_weight=0.5, balance_weight=0.1` for treatment.

| run | trainable | mean unique experts | Δ vs. baseline | perplexity | Δ vs. baseline |
|---|---|---|---|---|---|
| baseline | — | 20.761 | — | 7.119 | — |
| control (router-only) | router | 20.739 | -0.11% | within noise | ~0% |
| treatment, weight=0.1 (router-only) | router | 20.716 | -0.22% | 7.102 | -0.2% |
| treatment, weight=1.0 (router-only) | router | 20.716 | -0.22% | 7.298 | +2.5% |
| control (router+experts, stable LR) | router+experts | 20.784 | +0.11% | 6.792 | -4.6% |
| treatment (router+experts, stable LR) | router+experts | 20.727 | **-0.16%** | 7.088 | -0.4% |

## Conclusion

With training stable and quality preserved, the real, controlled locality
effect achievable so far is consistently tiny (roughly -0.1% to -0.5% in
mean unique experts touched) across router-only and router+experts scopes,
and across a 10x weight range. This held across three independent stable
configurations. The loss and training infrastructure are validated as
numerically correct; the bottleneck is the effect size, not a remaining
implementation bug.

Two candidate explanations for the small effect, not yet tested:

- **Loss design**: the current loss penalizes union/churn *within one
  sequence*. SPEC's actual premise is reuse *across requests in a serving
  session* (so a resident expert cache stays warm) — a within-sequence
  loss only approximates that, and may need reframing around a
  session/global expert-usage distribution instead.
- **Training scale**: only 2000 steps over 20% of a 2-file Wikitext-2
  slice was tried. Materially more data/steps was not ruled out.

## Status

Paused pending a decision on which of these (or method 1/3 instead) to
invest in next; not resumed as of this writeup.
