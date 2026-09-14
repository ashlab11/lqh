import torch

from lqh.experiments.shifted_router.distill import ShiftedGateDistillCollector, shifted_gate_distill_loss
from lqh.experiments.shifted_router.model import add_shifted_gates
from lqh.experiments.shifted_router.trace import ShiftedRouterTraceCollector, attach_shifted_router_trace
from tests.unit.test_shifted_router_hooks import FakeModel


def test_add_shifted_gates_copies_weights_and_freezes_everything_else() -> None:
    model = FakeModel(num_layers=2)

    enabled = add_shifted_gates(model)

    assert set(enabled) == {
        "layers.0.feed_forward.shifted_gate.weight",
        "layers.1.feed_forward.shifted_gate.weight",
    }
    for layer in model.layers:
        assert torch.equal(layer.feed_forward.shifted_gate.weight, layer.feed_forward.gate.weight)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (name in enabled)


def test_shifted_gate_distill_loss_decreases_with_gradient_steps() -> None:
    """A copy-initialized shifted_gate starts with zero distillation loss
    (it's identical to the real gate); perturb it and confirm a gradient
    step on the distillation loss moves it back toward agreement."""
    torch.manual_seed(0)
    model = FakeModel(num_layers=1)
    add_shifted_gates(model)

    collector = ShiftedGateDistillCollector()
    handles = collector.attach(model)
    try:
        model(torch.randn(2, 6, 6))
        loss_before, metrics_before = shifted_gate_distill_loss(collector.layer_captures)
    finally:
        for handle in handles:
            handle.remove()
    assert metrics_before["distill_layers"] == 1

    with torch.no_grad():
        model.layers[0].feed_forward.shifted_gate.weight.add_(torch.randn_like(model.layers[0].feed_forward.shifted_gate.weight))

    collector.clear()
    handles = collector.attach(model)
    try:
        model(torch.randn(2, 6, 6))
        loss_perturbed, _ = shifted_gate_distill_loss(collector.layer_captures)
    finally:
        for handle in handles:
            handle.remove()

    assert loss_perturbed.item() > loss_before.item()

    optimizer = torch.optim.SGD(model.layers[0].feed_forward.shifted_gate.parameters(), lr=0.5)
    for _ in range(20):
        collector.clear()
        handles = collector.attach(model)
        try:
            model(torch.randn(2, 6, 6))
            loss, _ = shifted_gate_distill_loss(collector.layer_captures)
        finally:
            for handle in handles:
                handle.remove()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    collector.clear()
    handles = collector.attach(model)
    try:
        model(torch.randn(2, 6, 6))
        loss_after, _ = shifted_gate_distill_loss(collector.layer_captures)
    finally:
        for handle in handles:
            handle.remove()
    assert loss_after.item() < loss_perturbed.item()


def test_trained_shifted_gate_is_evaluable_via_trace_with_gate_attr() -> None:
    """A freshly copy-initialized shifted_gate is bit-identical to the real
    gate, so tracing with gate_attr="shifted_gate" must give the exact same
    hit/wasted rate as the zero-shot gate_attr="gate" default -- confirming
    the trace hooks correctly read the requested attribute rather than
    always falling back to the real gate."""
    torch.manual_seed(1)
    model = FakeModel(num_layers=1)

    zero_shot_collector = ShiftedRouterTraceCollector()
    handles = attach_shifted_router_trace(model, zero_shot_collector, gate_attr="gate")
    inputs = torch.randn(1, 5, 6)
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    add_shifted_gates(model)
    shifted_collector = ShiftedRouterTraceCollector()
    handles = attach_shifted_router_trace(model, shifted_collector, gate_attr="shifted_gate")
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    assert len(zero_shot_collector.records) == len(shifted_collector.records) == 1
    assert shifted_collector.records[0].mean_hit_rate == zero_shot_collector.records[0].mean_hit_rate
    assert shifted_collector.records[0].mean_wasted_rate == zero_shot_collector.records[0].mean_wasted_rate
