"""SoftTopK: a differentiable top-k mask, per Hou et al. 2025 (IFPruning) eq. 3.

Given per-channel importance scores z and a target keep-count t, produces a
mask m where exactly t entries are nonzero (equal to a soft, differentiable
weight lambda) and the rest are exactly zero. Gradients flow through lambda
for the kept channels; the top-k selection itself is treated as a constant
w.r.t. the backward pass (standard straight-through practice for a
differentiable top-k).

We use lambda = t * softmax(z) as the normalization function g(.) (the paper
cites SoftTopK's use in CoLT5 (Ainslie et al. 2023) and Conditional Adapters
(Lei et al. 2023) without giving g(.) explicitly here) -- this is the same
"relaxed top-k membership probability" construction already validated in
lqh/experiments/shifted_router/loss.py's union-loss calibration fix, since
softmax shares logits' own rank order and sums to the target count.
"""

from __future__ import annotations

from typing import Any


def soft_topk_mask(scores: Any, keep_count: int) -> Any:
    """Return a mask with the same shape as `scores` (last dim = d_ffn).

    Exactly `keep_count` entries per row are nonzero, each equal to
    `keep_count * softmax(scores)` at that position; all other entries are
    exactly zero.
    """
    import torch

    if keep_count <= 0 or keep_count >= scores.shape[-1]:
        raise ValueError(f"keep_count must be in (0, d_ffn); got {keep_count} for d_ffn={scores.shape[-1]}")
    lam = keep_count * torch.softmax(scores, dim=-1)
    top_indices = torch.topk(lam, k=keep_count, dim=-1).indices
    keep = torch.zeros_like(lam, dtype=torch.bool).scatter_(-1, top_indices, True)
    return lam * keep
