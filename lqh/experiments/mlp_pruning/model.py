"""Inject a per-example, per-layer FFN mask into a dense Lfm2ForCausalLM.

Lfm2MLP computes w2(silu(w1(x)) * w3(x)); IFPruning masks the intermediate
activation silu(w1(x)) * w3(x) (shape [..., intermediate_size]) before w2, per
Hou et al. 2025 eq. 2. The mask is decided once per example from its
instruction (see predictor.py) and held fixed for every token in that
example's sequence -- so it's threaded in via a shared mutable context
object, not recomputed per token.
"""

from __future__ import annotations

import re
from typing import Any


def _layer_index(module_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if match is None:
        raise ValueError(f"Could not determine layer index from {module_name!r}")
    return int(match.group(1))


class PruningMaskContext:
    """Holds the current forward pass's mask: [batch, num_ffn_layers, d_ffn], or None (dense passthrough)."""

    def __init__(self) -> None:
        self.mask: Any = None


def patch_mlp_with_mask(model: Any, mask_context: PruningMaskContext) -> list[tuple[Any, Any]]:
    """Monkeypatch every Lfm2MLP to apply mask_context.mask[:, layer_idx, :]
    (broadcast over the sequence dim) to its intermediate activation, if set.

    Returns [(module, original_forward), ...] for reverting.
    """
    import types

    import torch.nn.functional as F

    patched: list[tuple[Any, Any]] = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "Lfm2MLP":
            continue
        layer_idx = _layer_index(module_name)

        def masked_forward(self: Any, x: Any, *, _layer_idx: int = layer_idx, _ctx: PruningMaskContext = mask_context) -> Any:
            gate = F.silu(self.w1(x)) * self.w3(x)
            if _ctx.mask is not None:
                gate = gate * _ctx.mask[:, _layer_idx, :].unsqueeze(1)
            return self.w2(gate)

        patched.append((module, module.forward))
        module.forward = types.MethodType(masked_forward, module)

    if not patched:
        raise ValueError("No Lfm2MLP modules found on the supplied model")
    return patched


def revert_mlp_patch(patched: list[tuple[Any, Any]]) -> None:
    for module, original_forward in patched:
        module.forward = original_forward


def count_ffn_layers(model: Any) -> int:
    return sum(1 for _name, module in model.named_modules() if module.__class__.__name__ == "Lfm2MLP")
