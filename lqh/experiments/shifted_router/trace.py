"""Measure SSD-prefetch feasibility for a "shifted router" on LFM2 MoE.

Method 3's proposal (see repo-level SPEC.md) moves each MoE layer's router
computation one layer earlier: instead of routing on the post-attention
hidden state that normally feeds a layer's MoE block, use that layer's own
*input* hidden state (i.e. the previous layer's output) -- giving a full
attention/conv layer's worth of lead time to prefetch the predicted experts
from SSD before they're actually needed.

This module is purely observational and requires no training: it runs the
*existing, untrained* router weights on both the real (post-attention) and
"early" (pre-attention) hidden states for every MoE layer, and compares the
two top-k selections. This measures whether the current router's own weights
already carry enough signal one layer early to be a useful prefetch oracle --
before investing in fine-tuning a shifted router specifically.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class SequenceLayerPrefetch:
    """Prefetch-feasibility statistics for one request sequence at one layer."""

    layer: int
    sequence_index: int
    tokens: int
    top_k: int
    mean_hit_rate: float
    mean_wasted_rate: float
    real_unique_experts: int
    shifted_unique_experts: int
    shifted_extra_unique_experts: int


def _layer_index(module_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if match is None:
        raise ValueError(f"Could not determine layer index from {module_name!r}")
    return int(match.group(1))


def summarize_prefetch(
    real_selected: list[list[int]],
    shifted_selected: list[list[int]],
    *,
    layer: int,
    sequence_index: int,
) -> SequenceLayerPrefetch:
    """Compare a layer's real vs. one-layer-early ("shifted") expert selections.

    Kept free of Torch so this math is unit-testable in a CPU-only LQH
    development environment, matching lqh/experiments/router_locality/trace.py.
    """
    if not real_selected:
        raise ValueError("real_selected must contain at least one token")
    if len(real_selected) != len(shifted_selected):
        raise ValueError("real_selected and shifted_selected must have the same token count")
    top_k = len(real_selected[0])
    if top_k == 0 or any(len(token) != top_k for token in real_selected + shifted_selected):
        raise ValueError("each token must have the same non-zero top-k width in both selections")

    hit_rates: list[float] = []
    wasted_rates: list[float] = []
    real_union: set[int] = set()
    shifted_union: set[int] = set()
    for real_token, shifted_token in zip(real_selected, shifted_selected):
        real_set = set(int(e) for e in real_token)
        shifted_set = set(int(e) for e in shifted_token)
        real_union |= real_set
        shifted_union |= shifted_set
        hit_rates.append(len(real_set & shifted_set) / len(real_set))
        wasted_rates.append(len(shifted_set - real_set) / len(shifted_set))

    return SequenceLayerPrefetch(
        layer=layer,
        sequence_index=sequence_index,
        tokens=len(real_selected),
        top_k=top_k,
        mean_hit_rate=sum(hit_rates) / len(hit_rates),
        mean_wasted_rate=sum(wasted_rates) / len(wasted_rates),
        real_unique_experts=len(real_union),
        shifted_unique_experts=len(shifted_union),
        shifted_extra_unique_experts=len(shifted_union - real_union),
    )


class ShiftedRouterTraceCollector:
    """Collect real-vs-shifted routing comparisons for one or more forwards."""

    def __init__(self) -> None:
        self.records: list[SequenceLayerPrefetch] = []

    def record(
        self,
        *,
        layer: int,
        batch_size: int,
        sequence_length: int,
        real_selected_experts: Any,
        shifted_selected_experts: Any,
    ) -> None:
        expected = batch_size * sequence_length
        if int(real_selected_experts.shape[0]) != expected or int(shifted_selected_experts.shape[0]) != expected:
            raise ValueError(
                "Router selection shape does not match owning MoE block: "
                f"real={real_selected_experts.shape[0]}, shifted={shifted_selected_experts.shape[0]}, "
                f"expected {expected} rows"
            )
        real_rows = real_selected_experts.detach().to("cpu").tolist()
        shifted_rows = shifted_selected_experts.detach().to("cpu").tolist()
        for sequence_index in range(batch_size):
            start = sequence_index * sequence_length
            stop = start + sequence_length
            self.records.append(
                summarize_prefetch(
                    real_rows[start:stop],
                    shifted_rows[start:stop],
                    layer=layer,
                    sequence_index=sequence_index,
                )
            )

    def clear(self) -> None:
        self.records.clear()

    def aggregate(self) -> dict[str, float | int]:
        """Return request-weighted metrics suitable for JSON/CSV reporting."""
        if not self.records:
            return {"sequence_layer_records": 0}
        return {
            "sequence_layer_records": len(self.records),
            "mean_hit_rate": sum(r.mean_hit_rate for r in self.records) / len(self.records),
            "mean_wasted_rate": sum(r.mean_wasted_rate for r in self.records) / len(self.records),
            "mean_real_unique_experts": sum(r.real_unique_experts for r in self.records) / len(self.records),
            "mean_shifted_unique_experts": sum(r.shifted_unique_experts for r in self.records) / len(self.records),
        }


def attach_shifted_router_trace(model: Any, collector: ShiftedRouterTraceCollector) -> list[Any]:
    """Attach non-invasive hooks comparing real vs. one-layer-early routing.

    For every MoE decoder layer, captures the layer's own input hidden state
    (pre-attention/conv) via a forward pre-hook on the decoder layer itself,
    then at the real router's forward hook, feeds that early hidden state
    through the *same* (untrained) gate + ffn_norm to get a "shifted"
    selection, and records real vs. shifted agreement. Returns hook handles;
    callers must remove them after the benchmark run.
    """
    handles: list[Any] = []
    for layer_name, layer in model.named_modules():
        if layer.__class__.__name__ != "Lfm2MoeDecoderLayer":
            continue
        block = getattr(layer, "feed_forward", None)
        if block is None or block.__class__.__name__ != "Lfm2MoeSparseMoeBlock":
            continue  # dense (non-MoE) layers have no router to shift
        layer_idx = _layer_index(layer_name)
        shape: dict[str, int] = {}
        state: dict[str, Any] = {}

        def capture_layer_input(
            _module: Any, inputs: tuple[Any, ...], *, _state: dict[str, Any] = state
        ) -> None:
            _state["early_hidden_states"] = inputs[0]

        def capture_block_shape(
            _module: Any, inputs: tuple[Any, ...], *, _shape: dict[str, int] = shape
        ) -> None:
            hidden_states = inputs[0]
            _shape["batch_size"] = int(hidden_states.shape[0])
            _shape["sequence_length"] = int(hidden_states.shape[1])

        def capture_router_output(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            _layer: int = layer_idx,
            _shape: dict[str, int] = shape,
            _block: Any = block,
            _decoder_layer: Any = layer,
            _state: dict[str, Any] = state,
        ) -> None:
            if "batch_size" not in _shape or "sequence_length" not in _shape:
                raise RuntimeError("MoE router executed before its block-shape hook")
            if "early_hidden_states" not in _state:
                raise RuntimeError("Decoder layer's input was not captured before its router ran")
            import torch

            if isinstance(output, tuple):
                real_selected = output[2]
            else:
                real_selected, _ = _block.route_tokens_to_experts(output)
            early_hidden_states = _state.pop("early_hidden_states")
            with torch.no_grad():
                # Call the gate's linear op directly (not _block.gate(...)):
                # calling the module itself would re-trigger this very
                # forward hook re-entrantly, since it's registered on
                # _block.gate.
                # Flatten to [B*S, H] before the gate, matching the real
                # path's own convention (Lfm2MoeSparseMoeBlock.forward
                # reshapes hidden_states the same way before its self.gate
                # call), so shifted_selected comes out [B*S, top_k] like
                # real_selected.
                normed_early = _decoder_layer.ffn_norm(early_hidden_states).reshape(-1, early_hidden_states.shape[-1])
                shifted_logits = torch.nn.functional.linear(
                    normed_early, _block.gate.weight, _block.gate.bias
                )
                shifted_selected, _ = _block.route_tokens_to_experts(shifted_logits)
            collector.record(
                layer=_layer,
                batch_size=_shape["batch_size"],
                sequence_length=_shape["sequence_length"],
                real_selected_experts=real_selected,
                shifted_selected_experts=shifted_selected,
            )

        handles.append(layer.register_forward_pre_hook(capture_layer_input))
        handles.append(block.register_forward_pre_hook(capture_block_shape))
        handles.append(block.gate.register_forward_hook(capture_router_output))
    if not handles:
        raise ValueError("No Lfm2MoeSparseMoeBlock modules found on the supplied model")
    return handles
