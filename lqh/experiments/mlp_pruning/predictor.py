"""Sparsity predictor: reads an instruction, predicts per-layer FFN masks.

Per Hou et al. 2025 (IFPruning) Sec 3.2: a small pretrained LM backbone
extracts the instruction's last-token hidden state, and a two-layer MLP head
maps that to per-layer, per-channel importance scores z in R^(L x d_ffn),
which soft_topk_mask.py turns into masks.
"""

from __future__ import annotations

from typing import Any


class SparsityPredictor:
    """Wraps a small causal LM backbone + mask-prediction head.

    Not an nn.Module subclass itself, since the backbone and head need
    independent optimizer/freezing treatment from the pruned target model --
    callers hold `.backbone` and `.head` directly.
    """

    def __init__(self, backbone: Any, num_ffn_layers: int, ffn_dim: int, mlp_hidden_dim: int = 128) -> None:
        import torch.nn as nn

        self.backbone = backbone
        self.num_ffn_layers = num_ffn_layers
        self.ffn_dim = ffn_dim
        backbone_hidden_size = backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(backbone_hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_ffn_layers * ffn_dim),
        )

    def to(self, *args: Any, **kwargs: Any) -> "SparsityPredictor":
        self.backbone = self.backbone.to(*args, **kwargs)
        self.head = self.head.to(*args, **kwargs)
        return self

    def parameters(self) -> Any:
        import itertools

        return itertools.chain(self.backbone.parameters(), self.head.parameters())

    def predict_scores(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return per-layer, per-channel importance scores z: [batch, num_ffn_layers, ffn_dim].

        Uses the hidden state of each sequence's own last non-padded token
        (per Hou et al.'s "hidden states of the last token x_n"), which is
        NOT simply index -1 whenever there's left-over padding. Assumes
        right-padding (attention_mask's valid tokens are a contiguous prefix
        of each row) -- callers must set tokenizer.padding_side="right".
        """
        import torch

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [batch, seq, backbone_hidden]
        last_token_index = attention_mask.sum(dim=1) - 1  # 0-indexed position of each row's last real token
        batch_index = torch.arange(last_token_index.shape[0], device=last_hidden.device)
        last_token_hidden = last_hidden[batch_index, last_token_index]  # [batch, backbone_hidden]
        scores = self.head(last_token_hidden)  # [batch, num_ffn_layers * ffn_dim]
        return scores.view(-1, self.num_ffn_layers, self.ffn_dim)

    def predict_mask(self, input_ids: Any, attention_mask: Any, keep_count: int) -> Any:
        from lqh.experiments.mlp_pruning.soft_topk import soft_topk_mask

        scores = self.predict_scores(input_ids, attention_mask)
        return soft_topk_mask(scores, keep_count)
