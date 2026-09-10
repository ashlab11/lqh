"""Add a small, separately-trainable shifted-router head to an LFM2 MoE model.

Rather than rewiring the model's actual forward pass to route on an earlier
hidden state (a real architecture change that would need validating against
the full model's output quality), this adds a *duplicate* gate --
``shifted_gate``, copy-initialized from the real (frozen) ``gate`` -- to
every MoE block. It is trained by distillation against the real router's own
decisions (see ``lqh/experiments/shifted_router/distill.py``) while every
other parameter, including the real gate and all experts, stays frozen: the
model's real forward pass and output quality are completely unaffected by
this training. Once a shifted_gate is good enough, deploying it means
literally moving the router (using shifted_gate's weights, fed the
one-layer-early hidden state) in place of the original -- but validating and
training it this way carries zero risk to the base model in the meantime.
"""

from __future__ import annotations

from typing import Any


def add_shifted_gates(model: Any) -> list[str]:
    """Add a trainable ``shifted_gate`` to every MoE block; freeze everything else.

    Returns the parameter names of the newly added shifted gates (the only
    parameters left trainable).
    """
    import torch
    import torch.nn as nn

    added_parameter_names: list[str] = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "Lfm2MoeSparseMoeBlock":
            continue
        gate = module.gate
        shifted_gate = nn.Linear(gate.in_features, gate.out_features, bias=gate.bias is not None)
        shifted_gate = shifted_gate.to(device=gate.weight.device, dtype=gate.weight.dtype)
        with torch.no_grad():
            shifted_gate.weight.copy_(gate.weight)
            if gate.bias is not None:
                shifted_gate.bias.copy_(gate.bias)
        module.add_module("shifted_gate", shifted_gate)
        added_parameter_names.append(f"{module_name}.shifted_gate.weight")
        if gate.bias is not None:
            added_parameter_names.append(f"{module_name}.shifted_gate.bias")

    if not added_parameter_names:
        raise ValueError("No Lfm2MoeSparseMoeBlock modules found on the supplied model")

    trainable = set(added_parameter_names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in trainable
    return added_parameter_names
