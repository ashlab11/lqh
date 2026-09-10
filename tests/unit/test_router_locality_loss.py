import torch

from lqh.experiments.router_locality.loss import LocalityLossConfig, locality_loss


def _one_hot_logits(batch: int, seq: int, num_experts: int, expert: int, magnitude: float = 10.0) -> torch.Tensor:
    logits = torch.zeros(batch, seq, num_experts)
    logits[..., expert] = magnitude
    return logits


def test_union_loss_is_lower_when_every_token_prefers_the_same_expert() -> None:
    config = LocalityLossConfig(union_weight=1.0, top_k=1)
    mask = torch.ones(1, 4, dtype=torch.long)

    concentrated = _one_hot_logits(1, 4, num_experts=4, expert=0)
    _, concentrated_metrics = locality_loss([(0, concentrated)], mask, config)

    spread = torch.stack(
        [_one_hot_logits(1, 1, 4, expert)[:, 0] for expert in range(4)], dim=1
    )
    _, spread_metrics = locality_loss([(0, spread)], mask, config)

    assert concentrated_metrics["router_expected_union"].item() < spread_metrics["router_expected_union"].item()


def test_union_loss_is_not_shrunk_by_appending_redundant_valid_tokens() -> None:
    """Regression test: a prior version divided the per-sequence union estimate
    by valid-token count, so padding a sequence with tokens that route to an
    *already-touched* expert made the loss collapse toward zero instead of
    staying flat/increasing. That silently made union_weight nearly inert
    during real training (confirmed empirically: 200 SFT steps with
    union_weight=0.1 moved routing locality by <0.3% vs. an untouched
    baseline)."""
    config = LocalityLossConfig(union_weight=1.0, top_k=1)
    num_experts = 4

    base_mask = torch.ones(1, num_experts, dtype=torch.long)
    base_logits = torch.stack(
        [_one_hot_logits(1, 1, num_experts, expert)[:, 0] for expert in range(num_experts)], dim=1
    )
    _, base_metrics = locality_loss([(0, base_logits)], base_mask, config)

    redundant = _one_hot_logits(1, 28, num_experts, expert=0)
    extended_logits = torch.cat([base_logits, redundant], dim=1)
    extended_mask = torch.ones(1, num_experts + 28, dtype=torch.long)
    _, extended_metrics = locality_loss([(0, extended_logits)], extended_mask, config)

    assert extended_metrics["router_expected_union"].item() >= base_metrics["router_expected_union"].item() - 1e-4


def test_locality_loss_combines_weighted_terms() -> None:
    config = LocalityLossConfig(union_weight=2.0, churn_weight=3.0, balance_weight=5.0, top_k=1)
    mask = torch.ones(1, 4, dtype=torch.long)
    logits = torch.stack(
        [_one_hot_logits(1, 1, 4, expert)[:, 0] for expert in range(4)], dim=1
    )

    total, metrics = locality_loss([(0, logits)], mask, config)

    expected = (
        config.union_weight * metrics["router_expected_union"]
        + config.churn_weight * metrics["router_churn"]
        + config.balance_weight * metrics["router_balance"]
    )
    assert torch.allclose(total.detach(), expected)
    assert metrics["router_layers"] == 1
