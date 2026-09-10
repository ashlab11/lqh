"""Differentiable sequence-locality losses for LFM2 MoE routers.

This module is opt-in: importing it neither patches a model nor changes LQH's
standard SFT/DPO/GRPO paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class LocalityLossConfig:
    union_weight: float = 0.0
    churn_weight: float = 0.0
    balance_weight: float = 0.0
    top_k: int = 4
    epsilon: float = 1e-8


def _layer_index(module_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if match is None:
        raise ValueError(f"Could not determine layer index from {module_name!r}")
    return int(match.group(1))


class RouterLogitCollector:
    """Retain differentiable router logits from the current model forward."""

    def __init__(self) -> None:
        self.layer_logits: list[tuple[int, Any]] = []

    def clear(self) -> None:
        self.layer_logits.clear()

    def attach(self, model: Any) -> list[Any]:
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

            def capture_router_logits(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: tuple[Any, Any, Any],
                *,
                _layer: int = layer,
                _shape: dict[str, int] = shape,
            ) -> None:
                if "batch_size" not in _shape or "sequence_length" not in _shape:
                    raise RuntimeError("MoE router executed before its block-shape hook")
                logits = output[0]
                self.layer_logits.append(
                    (_layer, logits.reshape(_shape["batch_size"], _shape["sequence_length"], -1))
                )

            handles.append(block.register_forward_pre_hook(capture_block_shape))
            handles.append(block.gate.register_forward_hook(capture_router_logits))
        if not handles:
            raise ValueError("No Lfm2MoeSparseMoeBlock modules found on the supplied model")
        return handles


def locality_loss(
    layer_logits: list[tuple[int, Any]], attention_mask: Any, config: LocalityLossConfig
) -> tuple[Any, dict[str, Any]]:
    """Return differentiable proxies for expert union, churn and balance."""
    if not layer_logits:
        raise ValueError("No router logits were captured during the model forward")
    import torch

    valid = attention_mask.to(dtype=torch.bool)
    valid_counts = valid.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
    union_terms: list[Any] = []
    churn_terms: list[Any] = []
    balance_terms: list[Any] = []
    for _layer, logits in layer_logits:
        if logits.shape[:2] != valid.shape:
            raise ValueError(
                "Router logits and attention mask disagree on batch/sequence shape: "
                f"{tuple(logits.shape[:2])} != {tuple(valid.shape)}"
            )
        scores = torch.sigmoid(logits)
        probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(config.epsilon)
        masked_probabilities = probabilities * valid.unsqueeze(-1)
        draws = valid_counts[:, None, None] * config.top_k
        expected_union = 1 - torch.pow((1 - probabilities).clamp_min(config.epsilon), draws)
        union_terms.append(expected_union.sum(dim=-1).div(valid_counts).mean())
        if probabilities.shape[1] > 1:
            adjacent_valid = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
            adjacent_delta = (probabilities[:, 1:] - probabilities[:, :-1]).abs()
            churn_terms.append((adjacent_delta * adjacent_valid).sum() / adjacent_valid.sum().clamp_min(1))
        else:
            churn_terms.append(logits.new_zeros(()))
        global_distribution = masked_probabilities.sum(dim=(0, 1))
        global_distribution = global_distribution / global_distribution.sum().clamp_min(config.epsilon)
        uniform = torch.full_like(global_distribution, 1 / global_distribution.numel())
        balance_terms.append(torch.mean((global_distribution - uniform).square()))
    union = torch.stack(union_terms).mean()
    churn = torch.stack(churn_terms).mean()
    balance = torch.stack(balance_terms).mean()
    total = config.union_weight * union + config.churn_weight * churn + config.balance_weight * balance
    return total, {
        "router_locality_loss": total.detach(),
        "router_expected_union": union.detach(),
        "router_churn": churn.detach(),
        "router_balance": balance.detach(),
        "router_layers": len(layer_logits),
    }


def enable_router_only_training(model: Any) -> list[str]:
    """Freeze an LFM2 MoE model except for custom router weights."""
    router_parameter_names = {
        f"{module_name}.weight"
        for module_name, module in model.named_modules()
        if module.__class__.__name__ == "Lfm2MoeTopKRouter"
    }
    if not router_parameter_names:
        raise ValueError("No Lfm2MoeTopKRouter modules found on the supplied model")
    enabled: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in router_parameter_names
        if parameter.requires_grad:
            enabled.append(name)
    if set(enabled) != router_parameter_names:
        raise RuntimeError(f"Could not enable all router parameters: {sorted(router_parameter_names - set(enabled))}")
    return enabled
