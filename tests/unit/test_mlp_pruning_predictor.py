from dataclasses import dataclass

import torch
import torch.nn as nn

from lqh.experiments.mlp_pruning.predictor import SparsityPredictor


@dataclass
class FakeConfig:
    hidden_size: int


@dataclass
class FakeBackboneOutput:
    hidden_states: tuple


class FakeBackbone(nn.Module):
    """Mimics a HF causal LM backbone: embeds token ids, returns
    output_hidden_states=True-style output with the last layer's states."""

    def __init__(self, hidden_size: int = 16, vocab_size: int = 50) -> None:
        super().__init__()
        self.config = FakeConfig(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool = False) -> FakeBackboneOutput:
        hidden = self.embed(input_ids)
        return FakeBackboneOutput(hidden_states=(hidden,))


def test_predict_scores_uses_last_real_token_not_last_padded_position() -> None:
    """With right-padding, row 0 has 2 real tokens + 2 pad tokens; the
    predictor must pool from position 1 (the real last token), not position 3
    (a pad position), or its output would depend on padding length."""
    torch.manual_seed(0)
    backbone = FakeBackbone(hidden_size=16, vocab_size=50)
    predictor = SparsityPredictor(backbone, num_ffn_layers=3, ffn_dim=8)

    input_ids = torch.tensor([[5, 7, 0, 0], [5, 7, 9, 11]])
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    scores = predictor.predict_scores(input_ids, attention_mask)
    assert scores.shape == (2, 3, 8)

    # Recompute manually: row 0's pooled hidden state should come from
    # position 1 (token id 7), not position 3 (padding, token id 0).
    expected_row0_hidden = backbone.embed(torch.tensor(7))
    expected_row0_scores = predictor.head(expected_row0_hidden).view(3, 8)
    assert torch.allclose(scores[0], expected_row0_scores, atol=1e-5)


def test_predict_mask_has_correct_shape_and_sparsity() -> None:
    torch.manual_seed(0)
    backbone = FakeBackbone(hidden_size=16, vocab_size=50)
    predictor = SparsityPredictor(backbone, num_ffn_layers=2, ffn_dim=10)

    input_ids = torch.tensor([[5, 7, 9]])
    attention_mask = torch.ones(1, 3, dtype=torch.long)

    mask = predictor.predict_mask(input_ids, attention_mask, keep_count=4)

    assert mask.shape == (1, 2, 10)
    nonzero_per_layer = (mask != 0).sum(dim=-1)
    assert torch.equal(nonzero_per_layer, torch.full_like(nonzero_per_layer, 4))
