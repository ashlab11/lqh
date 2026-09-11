import torch
import torch.nn as nn
import torch.nn.functional as F

from lqh.experiments.mlp_pruning.model import (
    PruningMaskContext,
    count_ffn_layers,
    patch_mlp_with_mask,
    revert_mlp_patch,
)


class Lfm2MLP(nn.Module):
    def __init__(self, hidden_dim: int = 6, intermediate_dim: int = 8) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_forward = Lfm2MLP()


class FakeModel(nn.Module):
    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer.feed_forward(x)
        return x


def test_count_ffn_layers() -> None:
    model = FakeModel(num_layers=3)
    assert count_ffn_layers(model) == 3


def test_patch_with_none_mask_is_exact_passthrough() -> None:
    torch.manual_seed(0)
    model = FakeModel(num_layers=2)
    inputs = torch.randn(2, 4, 6)

    with torch.no_grad():
        before = model(inputs)

    ctx = PruningMaskContext()  # mask stays None
    patched = patch_mlp_with_mask(model, ctx)
    try:
        with torch.no_grad():
            after = model(inputs)
    finally:
        revert_mlp_patch(patched)

    assert torch.equal(before, after)


def test_mask_zeroes_out_pruned_channels_effect() -> None:
    """Zeroing every channel via the mask must make the FFN branch a no-op
    (each layer's residual output equals its own input)."""
    torch.manual_seed(0)
    model = FakeModel(num_layers=2)
    inputs = torch.randn(2, 4, 6)

    ctx = PruningMaskContext()
    patched = patch_mlp_with_mask(model, ctx)
    try:
        ctx.mask = torch.zeros(2, 2, 8)  # [batch, num_layers, d_ffn], all pruned
        with torch.no_grad():
            output = model(inputs)
    finally:
        revert_mlp_patch(patched)

    assert torch.equal(output, inputs)


def test_mask_differs_per_batch_example() -> None:
    """The mask is per-example: two batch rows with different masks must
    produce different outputs even from identical inputs."""
    torch.manual_seed(0)
    model = FakeModel(num_layers=1)
    inputs = torch.randn(1, 3, 6).repeat(2, 1, 1)  # identical rows

    ctx = PruningMaskContext()
    patched = patch_mlp_with_mask(model, ctx)
    try:
        mask = torch.ones(2, 1, 8)
        mask[0, 0, :4] = 0.0  # row 0 prunes half the channels
        ctx.mask = mask
        with torch.no_grad():
            output = model(inputs)
    finally:
        revert_mlp_patch(patched)

    assert not torch.equal(output[0], output[1])


def test_mask_dtype_mismatch_is_cast_not_left_to_implicit_promotion() -> None:
    """Regression test: a mask produced by softmax/topk in one dtype (e.g.
    fp32) multiplied against bf16 activations crashed a real cluster run as
    a low-level ROCm hardware exception before reaching self.w2's matmul.
    A model in bf16 fed a fp32 mask must still run cleanly."""
    torch.manual_seed(0)
    model = FakeModel(num_layers=1).to(torch.bfloat16)
    inputs = torch.randn(1, 3, 6, dtype=torch.bfloat16)

    ctx = PruningMaskContext()
    patched = patch_mlp_with_mask(model, ctx)
    try:
        ctx.mask = torch.ones(1, 1, 8, dtype=torch.float32)  # deliberately mismatched dtype
        with torch.no_grad():
            output = model(inputs)
    finally:
        revert_mlp_patch(patched)

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()


def test_revert_restores_original_forward() -> None:
    torch.manual_seed(0)
    model = FakeModel(num_layers=1)
    inputs = torch.randn(1, 3, 6)
    with torch.no_grad():
        before = model(inputs)

    ctx = PruningMaskContext()
    patched = patch_mlp_with_mask(model, ctx)
    ctx.mask = torch.zeros(1, 1, 8)
    revert_mlp_patch(patched)

    with torch.no_grad():
        after = model(inputs)
    assert torch.equal(before, after)
