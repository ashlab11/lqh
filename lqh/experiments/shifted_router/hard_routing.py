"""Evaluate "hard" shifted routing: commit fully to the one-layer-early guess.

Unlike lqh/experiments/shifted_router/trace.py (which only *compares* the
early guess against the real decision, never changing what the model
actually computes), this monkeypatches each MoE decoder layer's forward so
the layer's *actual* expert computation uses the routing decided from its
own (pre-attention) input hidden state -- with no fallback or retry for
experts the real (post-attention) state would have picked instead. This is
the true SSD-prefetch scenario with zero tolerance for misses: whatever the
shifted gate guessed is exactly what runs, wrong or not.

Evaluation-only: this changes the forward computation for the duration it's
applied, not any weights. Always call revert_hard_shifted_routing (or use
the context manager) to restore normal behavior afterward.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def apply_hard_shifted_routing(model: Any, *, gate_attr: str = "gate") -> list[tuple[Any, Any]]:
    """Monkeypatch every MoE decoder layer to route on its own input hidden
    state via ``getattr(feed_forward, gate_attr)``, with no retrieval of any
    expert the real (post-attention) state would have selected instead.

    Returns ``[(layer, original_forward), ...]`` for ``revert_hard_shifted_routing``.
    """
    import types

    import torch

    patched: list[tuple[Any, Any]] = []
    for _layer_name, layer in model.named_modules():
        if layer.__class__.__name__ != "Lfm2MoeDecoderLayer":
            continue
        block = getattr(layer, "feed_forward", None)
        if block is None or block.__class__.__name__ != "Lfm2MoeSparseMoeBlock":
            continue  # dense (non-MoE) layers are untouched

        def hard_shifted_forward(
            self: Any,
            hidden_states: Any,
            position_embeddings: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            *,
            _block: Any = block,
            _gate_attr: str = gate_attr,
            **kwargs: Any,
        ) -> Any:
            residual = hidden_states
            batch, seq, hidden_dim = hidden_states.shape
            gate_module = getattr(_block, _gate_attr)
            early_normed = self.ffn_norm(hidden_states).reshape(-1, hidden_dim)
            early_logits = torch.nn.functional.linear(early_normed, gate_module.weight, gate_module.bias)
            selected_experts, routing_weights = _block.route_tokens_to_experts(early_logits)

            if self.is_attention_layer:
                hidden_states, _ = self.self_attn(
                    hidden_states=self.operator_norm(hidden_states),
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            else:
                hidden_states = self.conv(
                    hidden_states=self.operator_norm(hidden_states),
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                )
            hidden_states = hidden_states + residual

            post_attn_normed = self.ffn_norm(hidden_states).reshape(-1, hidden_dim)
            expert_output = _block.experts(post_attn_normed, selected_experts, routing_weights)
            hidden_states = hidden_states + expert_output.view(batch, seq, hidden_dim)
            return hidden_states

        patched.append((layer, layer.forward))
        layer.forward = types.MethodType(hard_shifted_forward, layer)

    if not patched:
        raise ValueError("No Lfm2MoeSparseMoeBlock modules found on the supplied model")
    return patched


def revert_hard_shifted_routing(patched: list[tuple[Any, Any]]) -> None:
    """Undo apply_hard_shifted_routing, restoring each layer's real forward."""
    for layer, original_forward in patched:
        layer.forward = original_forward


@contextmanager
def hard_shifted_routing(model: Any, *, gate_attr: str = "gate") -> Iterator[None]:
    patched = apply_hard_shifted_routing(model, gate_attr=gate_attr)
    try:
        yield
    finally:
        revert_hard_shifted_routing(patched)
