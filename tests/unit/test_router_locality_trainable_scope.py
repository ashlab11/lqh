import torch
import torch.nn as nn

from lqh.experiments.router_locality.loss import (
    enable_router_and_expert_training,
    enable_router_only_training,
)


class Lfm2MoeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, 8, 4))
        self.down_proj = nn.Parameter(torch.randn(4, 4, 8))


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = Lfm2MoeExperts()


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.feed_forward = Lfm2MoeSparseMoeBlock()


class FakeModel(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer() for _ in range(num_layers)])


def test_router_only_training_freezes_experts_and_attention() -> None:
    model = FakeModel()

    enabled = enable_router_only_training(model)

    assert set(enabled) == {"layers.0.feed_forward.gate.weight", "layers.1.feed_forward.gate.weight"}
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (name in enabled)


def test_router_and_expert_training_also_unfreezes_expert_ffns() -> None:
    """Regression test: router-only training empirically capped the locality
    loss's effect (10x union_weight gave the same tiny locality shift but a
    real perplexity cost), because the frozen expert bank can't specialize
    around the router's new choices. This matches the original spec of
    SFTing router+experts together, not router-only."""
    model = FakeModel()

    enabled = enable_router_and_expert_training(model)

    for layer in range(2):
        assert f"layers.{layer}.feed_forward.gate.weight" in enabled
        assert f"layers.{layer}.feed_forward.experts.gate_up_proj" in enabled
        assert f"layers.{layer}.feed_forward.experts.down_proj" in enabled
    assert "layers.0.attn.weight" not in enabled
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (name in enabled)
