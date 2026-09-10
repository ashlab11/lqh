"""TRL trainer wrapper for opt-in MoE router-locality post-training."""

from __future__ import annotations

from typing import Any

from .loss import LocalityLossConfig, RouterLogitCollector, locality_loss


class RouterLocalitySFTTrainer:
    """Factory namespace to keep TRL imports out of normal LQH imports."""

    @staticmethod
    def build(*, locality: LocalityLossConfig, collector: RouterLogitCollector) -> type[Any]:
        from trl import SFTTrainer

        class _RouterLocalitySFTTrainer(SFTTrainer):
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                **kwargs: Any,
            ) -> Any:
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    raise ValueError("router-locality training requires attention_mask")
                collector.clear()
                base_loss, outputs = super().compute_loss(
                    model, inputs, return_outputs=True, **kwargs
                )
                auxiliary_loss, metrics = locality_loss(
                    collector.layer_logits, attention_mask, locality
                )
                total_loss = base_loss + auxiliary_loss
                if model.training:
                    self.log({key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()})
                return (total_loss, outputs) if return_outputs else total_loss

        return _RouterLocalitySFTTrainer
