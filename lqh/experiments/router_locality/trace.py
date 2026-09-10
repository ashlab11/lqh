"""Capture and summarize LFM2-MoE routing decisions without changing outputs.

The LFM2 MoE router returns ``(logits, weights, selected_experts)``.  This
module attaches forward hooks to the router and its owning sparse-MoE block,
copies only the selected expert IDs and top-k weights to CPU, and calculates
request-locality metrics.  It deliberately does not retain router logits yet:
the observation milestone needs low-overhead traces, while a later training
loss will consume logits in-process.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class SequenceLayerRouting:
    """Routing statistics for one request sequence at one MoE layer."""

    layer: int
    sequence_index: int
    tokens: int
    top_k: int
    unique_experts: int
    expert_selection_counts: dict[int, int]
    mean_adjacent_jaccard_distance: float


def _layer_index(module_name: str) -> int:
    """Extract the decoder-layer index from a Transformers module path."""
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if match is None:
        raise ValueError(f"Could not determine layer index from {module_name!r}")
    return int(match.group(1))


def _jaccard_distance(left: set[int], right: set[int]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def summarize_selected_experts(
    selected_experts: list[list[int]], *, layer: int, sequence_index: int
) -> SequenceLayerRouting:
    """Summarize a ``[tokens][top_k]`` expert-selection trace.

    Kept free of Torch so locality math is unit-testable in a CPU-only LQH
    development environment.
    """
    if not selected_experts:
        raise ValueError("selected_experts must contain at least one token")
    top_k = len(selected_experts[0])
    if top_k == 0 or any(len(token) != top_k for token in selected_experts):
        raise ValueError("each token must have the same non-zero top-k width")

    token_sets = [set(int(expert) for expert in token) for token in selected_experts]
    counts = Counter(expert for token in selected_experts for expert in token)
    churn = [
        _jaccard_distance(left, right)
        for left, right in zip(token_sets, token_sets[1:])
    ]
    return SequenceLayerRouting(
        layer=layer,
        sequence_index=sequence_index,
        tokens=len(selected_experts),
        top_k=top_k,
        unique_experts=len(counts),
        expert_selection_counts=dict(sorted(counts.items())),
        mean_adjacent_jaccard_distance=sum(churn) / len(churn) if churn else 0.0,
    )


class RouterTraceCollector:
    """Collect routing summaries for one or more model forward passes."""

    def __init__(self) -> None:
        self.records: list[SequenceLayerRouting] = []

    def record(
        self,
        *,
        layer: int,
        batch_size: int,
        sequence_length: int,
        selected_experts: Any,
    ) -> None:
        """Store routing metrics from LFM2's flattened ``[B*S, top_k]`` tensor."""
        expected = batch_size * sequence_length
        if int(selected_experts.shape[0]) != expected:
            raise ValueError(
                "Router selection shape does not match owning MoE block: "
                f"got {selected_experts.shape[0]} rows, expected {expected}"
            )
        rows = selected_experts.detach().to("cpu").tolist()
        for sequence_index in range(batch_size):
            start = sequence_index * sequence_length
            stop = start + sequence_length
            self.records.append(
                summarize_selected_experts(
                    rows[start:stop], layer=layer, sequence_index=sequence_index
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
            "mean_unique_experts": sum(r.unique_experts for r in self.records)
            / len(self.records),
            "mean_adjacent_jaccard_distance": sum(
                r.mean_adjacent_jaccard_distance for r in self.records
            )
            / len(self.records),
        }


def attach_lfm2_moe_router_trace(model: Any, collector: RouterTraceCollector) -> list[Any]:
    """Attach non-invasive hooks to every Transformers ``Lfm2MoeSparseMoeBlock``.

    Returns hook handles; callers must remove them after the benchmark run.
    The explicit class-name check avoids silently tracing unrelated modules that
    happen to expose a ``gate`` attribute.
    """
    handles: list[Any] = []
    for block_name, block in model.named_modules():
        if block.__class__.__name__ != "Lfm2MoeSparseMoeBlock":
            continue
        layer = _layer_index(block_name)
        shape: dict[str, int] = {}

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
            _layer: int = layer,
            _shape: dict[str, int] = shape,
            _block: Any = block,
        ) -> None:
            if "batch_size" not in _shape or "sequence_length" not in _shape:
                raise RuntimeError("MoE router executed before its block-shape hook")
            if isinstance(output, tuple):
                selected_experts = output[2]
            else:
                selected_experts, _weights = _block.route_tokens_to_experts(output)
            collector.record(
                layer=_layer,
                batch_size=_shape["batch_size"],
                sequence_length=_shape["sequence_length"],
                selected_experts=selected_experts,
            )

        handles.append(block.register_forward_pre_hook(capture_block_shape))
        handles.append(block.gate.register_forward_hook(capture_router_output))
    if not handles:
        raise ValueError("No Lfm2MoeSparseMoeBlock modules found on the supplied model")
    return handles
