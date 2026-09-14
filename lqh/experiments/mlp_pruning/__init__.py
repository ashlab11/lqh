"""Instruction-following MLP pruning (Apple's IFPruning) for a dense LFM2 model.

See https://arxiv.org/abs/2501.02086 (Hou et al., "Instruction-Following
Pruning for Large Language Models"). A small sparsity predictor reads the
user instruction and predicts, per FFN layer, which intermediate-dimension
channels to keep; the same selection is then used for the entire sequence's
prefill+decode (no re-prediction per token, unlike MoE).
"""
