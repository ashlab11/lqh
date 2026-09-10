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
