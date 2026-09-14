"""Distill each MoE layer's real router decision into its shifted_gate.

Trains only the ``shifted_gate`` added by ``model.add_shifted_gates`` (every
other parameter, including the real ``gate``, is frozen -- see model.py's
docstring for why). The distillation target is the real router's own raw
sigmoid scores for the *same forward pass*: this is pure self-distillation
and needs no labels, so training data is just unlabeled text.
"""

from __future__ import annotations

from typing import Any


class ShiftedGateDistillCollector:
    """Capture (real_logits, early_hidden_states, decoder_layer, block) per MoE layer."""

    def __init__(self) -> None:
        self.layer_captures: list[tuple[int, Any, Any, Any, Any]] = []

    def clear(self) -> None:
        self.layer_captures.clear()

    def attach(self, model: Any) -> list[Any]:
        import re

        def layer_index(module_name: str) -> int:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
            if match is None:
                raise ValueError(f"Could not determine layer index from {module_name!r}")
            return int(match.group(1))

        handles: list[Any] = []
        for layer_name, layer in model.named_modules():
            if layer.__class__.__name__ != "Lfm2MoeDecoderLayer":
                continue
            block = getattr(layer, "feed_forward", None)
            if block is None or block.__class__.__name__ != "Lfm2MoeSparseMoeBlock":
                continue
            layer_idx = layer_index(layer_name)
            state: dict[str, Any] = {}

            def capture_layer_input(
                _module: Any, inputs: tuple[Any, ...], *, _state: dict[str, Any] = state
            ) -> None:
                _state["early_hidden_states"] = inputs[0]

            def capture_real_logits(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                _layer: int = layer_idx,
                _block: Any = block,
                _decoder_layer: Any = layer,
                _state: dict[str, Any] = state,
            ) -> None:
                if "early_hidden_states" not in _state:
                    raise RuntimeError("Decoder layer's input was not captured before its router ran")
                real_logits = output[0] if isinstance(output, tuple) else output
                early_hidden_states = _state.pop("early_hidden_states")
                self.layer_captures.append((_layer, real_logits, early_hidden_states, _decoder_layer, _block))

            handles.append(layer.register_forward_pre_hook(capture_layer_input))
            handles.append(block.gate.register_forward_hook(capture_real_logits))
        if not handles:
            raise ValueError("No Lfm2MoeSparseMoeBlock modules found on the supplied model")
        return handles


def shifted_gate_distill_loss(layer_captures: list[tuple[int, Any, Any, Any, Any]]) -> tuple[Any, dict[str, Any]]:
    """Binary-cross-entropy distillation loss: shifted_gate's sigmoid scores
    (from the early hidden state) toward the real gate's own sigmoid scores
    (from the same forward pass, detached -- a fixed, stable target since the
    real gate is frozen throughout this training)."""
    if not layer_captures:
        raise ValueError("No router captures to distill from")
    import torch
    import torch.nn.functional as F

    losses: list[Any] = []
    for _layer, real_logits, early_hidden_states, decoder_layer, block in layer_captures:
        normed_early = decoder_layer.ffn_norm(early_hidden_states).reshape(-1, early_hidden_states.shape[-1])
        shifted_logits = F.linear(normed_early, block.shifted_gate.weight, block.shifted_gate.bias)
        target = real_logits.detach().sigmoid()
        losses.append(F.binary_cross_entropy_with_logits(shifted_logits, target))
    total = torch.stack(losses).mean()
    return total, {"shifted_gate_distill_loss": total.detach(), "distill_layers": len(layer_captures)}
