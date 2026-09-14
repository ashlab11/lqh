# Memory-locality post-training for LFM MoE models

## Objective

Develop and evaluate post-training and inference techniques that reduce the
host-RAM working set needed to serve Liquid Foundation Models. The first
implemented technique is sequence-local MoE routing: fine-tune an LFM2.5
8B-A1B base model so that tokens in a request reuse a small, stable set of
experts at each MoE layer without materially degrading language-model quality.

## First milestone: trace before training

Before changing a loss or model, run a fixed, reproducible prompt suite through
the unmodified MoE model and capture routing observations for every MoE layer
and generated token. The trace must support calculation of:

- selected expert IDs and routing weights;
- unique experts per sequence and per layer;
- adjacent-token expert-set churn;
- per-expert load distribution and capacity/drop statistics, if exposed by the
  model;
- trace metadata: model revision, tokenizer, precision, prompt ID, seed,
  prompt length, and generation length.

No model weights, router behavior, or standard LQH training paths change in
this milestone.

## Locality-loss experiment

After the baseline trace is validated, add an opt-in `memory_locality` training
path. It must combine language-model loss with differentiable routing-locality
terms calculated from pre-top-k routing probabilities:

1. discourage large per-sequence expert unions;
2. discourage routing distribution changes across adjacent tokens;
3. retain a batch-level expert-load-balancing constraint.

Start with router-only adaptation. Compare the unmodified model, router-only
language-loss adaptation, and locality-loss ablations at matched training data,
seed, context lengths, and decoding settings.

## Evaluation

Report quality relative to the untouched 8B-A1B baseline, plus routing
locality. Later phases add an SSD-backed expert cache to report resident RAM,
SSD bytes read, cache-hit rate, prefill latency, and decode latency. Cold and
warm cache measurements remain separate.

## Non-goals for the first milestone

- No SSD streaming runtime or shifted-router architecture.
- No dense 2.6B pruning implementation.
- No managed-cloud custom training: custom research runs execute on the
  cluster/local editable fork.

## Acceptance criteria

The baseline trace is schema-versioned, reproducible from a saved prompt suite,
and has unit tests using a small fake MoE model. Its aggregate report makes it
possible to rank prompts/layers by unique-expert count and routing churn.

## Second milestone: shifted-router prefetch feasibility (method 3)

After method 2's SFT results were logged (see
`lqh/experiments/router_locality/RESULTS.md`), the project moved to method 3:
moving each MoE layer's router computation one layer earlier so its output
can prefetch experts from SSD ahead of need, without running any router
twice.

First step is observation-only, mirroring the method 2 milestone above: for
every MoE decoder layer, compare the real (post-attention) top-k expert
selection against what the *same, untrained* router weights would pick if
fed that layer's own input hidden state instead (one layer early, after the
same `ffn_norm`). No training or weight changes in this milestone.

Metrics (`lqh/experiments/shifted_router/trace.py`):

- per-token hit rate: fraction of the real top-k experts that the one-layer-
  early ("shifted") guess already includes -- this is the resident-cache hit
  rate a naive prefetch-from-the-shifted-guess policy would achieve;
- per-token wasted rate: fraction of the shifted guess not actually needed --
  SSD bandwidth spent prefetching experts the real routing didn't select;
- per-sequence/layer real vs. shifted unique-expert-set sizes.

If the untrained router's own weights already predict real routing well one
layer early, an SSD-prefetch runtime may need no fine-tuning at all before a
runtime prototype. If not, the next step is fine-tuning a shifted router
specifically (reusing method 2's training infrastructure) before building
the runtime.
