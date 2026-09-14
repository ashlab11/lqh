import torch
import torch.nn as nn

from lqh.experiments.shifted_router.trace import ShiftedRouterTraceCollector, attach_shifted_router_trace


class Lfm2MoeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, 8, 6))
        self.down_proj = nn.Parameter(torch.randn(4, 6, 8))


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.top_k = 2
        self.gate = nn.Linear(6, 4, bias=False)
        self.experts = Lfm2MoeExperts()

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = router_logits.sigmoid()
        weights, indices = torch.topk(routing_weights, k=self.top_k, dim=-1)
        return indices, weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)
        logits = self.gate(flat)
        _selected, weights = self.route_tokens_to_experts(logits)
        return flat.view(batch, seq, hidden), weights


class Lfm2MoeRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Lfm2MoeDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn_norm = Lfm2MoeRMSNorm(6)
        self.operator_norm = Lfm2MoeRMSNorm(6)
        self.feed_forward = Lfm2MoeSparseMoeBlock()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        attn_out = hidden_states * 1.5  # stand-in for attention/conv
        hidden_states = attn_out + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))[0]
        return hidden_states


class FakeModel(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Lfm2MoeDecoderLayer() for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_attach_shifted_router_trace_does_not_reenter_the_gate_hook() -> None:
    """Regression test: computing the shifted prediction by calling
    _block.gate(...) inside the gate's own forward hook re-triggers that
    same hook recursively (PyTorch fires forward hooks on every call to the
    hooked module, including one made from inside another hook), which
    popped "early_hidden_states" from state twice and crashed a real cluster
    run (RuntimeError: Decoder layer's input was not captured before its
    router ran). Must use a direct functional call instead."""
    model = FakeModel(num_layers=2)
    collector = ShiftedRouterTraceCollector()
    handles = attach_shifted_router_trace(model, collector)

    try:
        model(torch.randn(1, 5, 6))
    finally:
        for handle in handles:
            handle.remove()

    assert len(collector.records) == 2  # one per MoE layer
    for record in collector.records:
        assert record.tokens == 5
        assert record.top_k == 2


def test_shifted_prediction_uses_flattened_batch_seq_like_the_real_path() -> None:
    """Regression test: the real router's output comes out [B*S, top_k]
    (Lfm2MoeSparseMoeBlock.forward flattens hidden_states before its gate
    call), so the shifted computation must flatten the same way or
    collector.record's shape check fails."""
    model = FakeModel(num_layers=1)
    collector = ShiftedRouterTraceCollector()
    handles = attach_shifted_router_trace(model, collector)

    try:
        model(torch.randn(3, 7, 6))  # batch_size > 1 exercises the reshape
    finally:
        for handle in handles:
            handle.remove()

    assert len(collector.records) == 3  # one per sequence in the batch
    for record in collector.records:
        assert record.tokens == 7
