import pytest
import torch

from lqh.experiments.mlp_pruning.soft_topk import soft_topk_mask


def test_soft_topk_mask_keeps_exactly_keep_count_nonzero_entries() -> None:
    torch.manual_seed(0)
    scores = torch.randn(3, 2, 20)  # [batch, layers, d_ffn]

    mask = soft_topk_mask(scores, keep_count=5)

    nonzero_counts = (mask != 0).sum(dim=-1)
    assert torch.equal(nonzero_counts, torch.full_like(nonzero_counts, 5))


def test_soft_topk_mask_selects_highest_scoring_channels() -> None:
    scores = torch.zeros(1, 8)
    scores[0, [1, 3, 5]] = 10.0  # three channels should dominate

    mask = soft_topk_mask(scores, keep_count=3)

    kept = (mask[0] != 0).nonzero(as_tuple=True)[0].tolist()
    assert sorted(kept) == [1, 3, 5]


def test_soft_topk_mask_is_differentiable_toward_kept_channels() -> None:
    torch.manual_seed(0)
    scores = torch.randn(1, 10, requires_grad=True)

    mask = soft_topk_mask(scores, keep_count=4)
    mask.sum().backward()

    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0


def test_soft_topk_mask_rejects_out_of_range_keep_count() -> None:
    scores = torch.randn(1, 10)
    with pytest.raises(ValueError):
        soft_topk_mask(scores, keep_count=0)
    with pytest.raises(ValueError):
        soft_topk_mask(scores, keep_count=10)
