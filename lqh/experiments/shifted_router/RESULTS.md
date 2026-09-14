# Method 3 (shifted router) — hard-commit results

## The question

If experts are chosen *literally* from the previous layer's signal — no
retrieval, no fallback for whichever experts the real (post-attention) state
would have picked instead — how good is the model, and how good is the
prefetch signal itself?

## Setup

`lqh/experiments/shifted_router/hard_routing.py` monkeypatches every MoE
decoder layer so its real expert computation uses the routing decided from
the layer's own *input* hidden state (one layer early), with zero tolerance
for misses — whatever the shifted gate guesses is exactly what runs.
Evaluation-only; no weights change. Tested against `LiquidAI/LFM2.5-8B-A1B-Base`,
held-out Wikitext-2 test-split perplexity, same 4-prompt trace suite as the
earlier baseline for hit/wasted rate.

## Results

| routing | perplexity | Δ vs. real routing | mean hit rate | mean wasted rate |
|---|---|---|---|---|
| real (baseline) | 7.119 | — | — | — |
| hard-shifted, zero-shot (untrained gate) | 7.615 | **+7.0%** | 0.860 | 0.140 |
| hard-shifted, small SFT (2000-step distilled shifted_gate) | 7.331 | **+3.0%** | 0.884 | 0.116 |

The small SFT closes **57%** of the zero-shot quality gap (7.615 → 7.331,
vs. baseline 7.119), for ~150s of training on one GPU and a few-MB delta.

## Reading this

- Committing hard with no retrieval has a real, measurable cost even with
  the current 86-88% hit rate: missing ~12-14% of the correct experts per
  layer compounds across 22 layers into a perplexity increase big enough to
  matter (+3-7%), not something to ignore.
- The gap between hit-rate and perplexity cost isn't 1:1 — a wrong expert
  doesn't uniformly bomb output quality, but it's not free either.
- SFT genuinely helps quality here, not just the hit-rate proxy: this is the
  first result in the whole project that measures the *actual* end-to-end
  cost of a MoE-locality idea (as opposed to expert-usage statistics), and
  it moved in the direction the hit-rate numbers predicted.
- A real SSD-prefetch runtime wouldn't need to commit this hard — a miss
  there just costs a slower on-demand fetch, not a wrong expert. These
  numbers are the worst case (zero retrieval), useful as a lower bound: an
  actual runtime should do *better* than either row here, not worse.
