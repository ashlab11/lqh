import torch
import torch.nn as nn

from lqh.experiments.shifted_router.hard_routing import (
    apply_hard_shifted_routing,
    hard_shifted_routing,
    revert_hard_shifted_routing,
)


class FakeExperts(nn.Module):
    def __init__(self, num_experts: int = 4, hidden_dim: int = 6, intermediate_dim: int = 8) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        # Deliberately simple (not a faithful MoE compute) -- just needs to
        # depend on which experts/weights were selected, so different
        # routing decisions produce different outputs.
        num_experts = self.gate_up_proj.shape[0]
        one_hot = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts).to(hidden_states.dtype)
        weighted = (one_hot * top_k_weights.unsqueeze(-1)).sum(dim=1)  # [tokens, num_experts]
        expert_bias = self.down_proj.mean(dim=(1, 2))  # [num_experts], a per-expert scalar signature
        return hidden_states + (weighted * expert_bias).sum(dim=-1, keepdim=True)


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self, hidden_dim: int = 6, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = FakeExperts(num_experts=num_experts, hidden_dim=hidden_dim)

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = router_logits.sigmoid()
        weights, indices = torch.topk(routing_weights, k=self.top_k, dim=-1)
        return indices, weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)
        logits = self.gate(flat)
        selected, weights = self.route_tokens_to_experts(logits)
        return self.experts(flat, selected, weights).view(batch, seq, hidden), weights


class Lfm2MoeRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class FakeAttention(nn.Module):
    def __init__(self, hidden_dim: int = 6, scale: float = 1.5) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> tuple[torch.Tensor, None]:
        return hidden_states * self.scale, None


class Lfm2MoeDecoderLayer(nn.Module):
    """Mirrors the real Lfm2MoeDecoderLayer's forward exactly (branching on
    is_attention_layer, calling self_attn/operator_norm/ffn_norm the same
    way) so the hard-routing patch -- which replicates that same structure
    -- can be tested faithfully."""

    def __init__(self, hidden_dim: int = 6, attention_scale: float = 1.5) -> None:
        super().__init__()
        self.is_attention_layer = True
        self.self_attn = FakeAttention(hidden_dim, scale=attention_scale)
        self.ffn_norm = Lfm2MoeRMSNorm(hidden_dim)
        self.operator_norm = Lfm2MoeRMSNorm(hidden_dim)
        self.feed_forward = Lfm2MoeSparseMoeBlock(hidden_dim=hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.operator_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))[0]
        return hidden_states


class FakeModel(nn.Module):
    def __init__(self, num_layers: int = 2, attention_scale: float = 1.5) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Lfm2MoeDecoderLayer(attention_scale=attention_scale) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_hard_shifted_routing_matches_real_output_when_early_equals_post_attention() -> None:
    """When attention_scale=0, a layer's input and post-attention state are
    identical -- so routing on the "early" signal is routing on the exact
    same state the real router would have used, and hard-shifted output
    must match the unpatched model's output exactly. This pins down that
    the patch's residual/reshape/expert-call plumbing is correct,
    independent of whether the early signal is actually informative."""
    torch.manual_seed(0)
    model = FakeModel(num_layers=2, attention_scale=0.0)
    inputs = torch.randn(2, 5, 6)

    with torch.no_grad():
        real_output = model(inputs)

    patched = apply_hard_shifted_routing(model, gate_attr="gate")
    try:
        with torch.no_grad():
            patched_output = model(inputs)
    finally:
        revert_hard_shifted_routing(patched)

    assert torch.equal(real_output, patched_output)


def test_hard_shifted_routing_reverts_cleanly() -> None:
    model = FakeModel(num_layers=1)
    inputs = torch.randn(1, 4, 6)
    with torch.no_grad():
        before = model(inputs)

    with hard_shifted_routing(model, gate_attr="gate"):
        with torch.no_grad():
            model(inputs)  # patched forward runs without error

    with torch.no_grad():
        after = model(inputs)
    assert torch.equal(before, after)


def test_hard_shifted_routing_changes_output_when_early_differs_from_real() -> None:
    """Sanity check: when attention actually perturbs the hidden state (the
    normal case), routing on the early vs. real signal generally picks a
    different expert set, so the patched output should differ from the
    real one -- confirming the patch isn't silently a no-op."""
    torch.manual_seed(0)
    model = FakeModel(num_layers=1, attention_scale=1.5)
    inputs = torch.randn(1, 8, 6)

    with torch.no_grad():
        real_output = model(inputs)

    patched = apply_hard_shifted_routing(model, gate_attr="gate")
    try:
        with torch.no_grad():
            patched_output = model(inputs)
    finally:
        revert_hard_shifted_routing(patched)

    assert not torch.equal(real_output, patched_output)
