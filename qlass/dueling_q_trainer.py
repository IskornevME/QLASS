"""Trainer for candidate-normalized parametric-action DuelingQNet.

This trainer is intentionally small. The model already knows how to compute
its own loss from the grouped dueling batch:
    value_input_ids / value_attention_mask
    adv_input_ids / adv_attention_mask
    candidate_targets / candidate_mask

So the trainer's main job is to:
  1. pass the whole batch to DuelingQNet.forward();
  2. return outputs["loss"] to HuggingFace Trainer;
  3. surface useful auxiliary metrics such as q_loss/value_loss.
"""
from typing import Any, Dict, Optional, Tuple, Union

import os
import torch
from torch import nn
from transformers import Trainer


def _to_float(value: Any) -> Optional[float]:
    """Convert a scalar tensor / numeric value to a Python float for logging."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class DuelingQTrainer(Trainer):
    """Minimal Trainer subclass for DuelingQNet.

    DuelingQNet.forward() returns a dict-like object with at least:
        outputs["loss"]

    and, during training/evaluation with labels:
        outputs["q_loss"]
        outputs["value_loss"]
        outputs["value_target"]
        outputs["q_values"]
        outputs["advantages"]
        outputs["advantage_mean"]

    This trainer does not recompute the loss. It only delegates the full grouped batch to the model and logs detached auxiliary diagnostics.
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        """Compute loss by delegating to DuelingQNet.forward().
        """
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            if "loss" not in outputs:
                raise ValueError(
                    "DuelingQNet output does not contain 'loss'. "
                    "Make sure candidate_targets are present in the batch."
                )
            loss = outputs["loss"]
        else:
            # Defensive fallback for tuple-style outputs.
            loss = outputs[0]

        # Store auxiliary metrics. They will be merged into the next regular Trainer log event by overriding log().
        # This avoids calling self.log() on every micro-step and keeps logging_steps behavior intact.
        if isinstance(outputs, dict):
            self._last_dueling_metrics = self._extract_dueling_metrics(outputs)

        return (loss, outputs) if return_outputs else loss

    def _extract_dueling_metrics(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """Collect detached scalar diagnostics from DuelingQNet outputs."""
        metrics: Dict[str, float] = {}

        for src_key, log_key in [
            ("q_loss", "dueling/q_loss"),
            ("value_loss", "dueling/value_loss"),
            ("value_target", "dueling/value_target"),
            ("advantage_mean", "dueling/advantage_mean"),
            ("value", "dueling/value_pred"),
        ]:
            val = _to_float(outputs.get(src_key))
            if val is not None:
                metrics[log_key] = val

        q_values = outputs.get("q_values")
        if isinstance(q_values, torch.Tensor) and q_values.numel() > 0:
            q = q_values.detach().float()
            metrics["dueling/q_pred_mean"] = float(q.mean().cpu().item())
            metrics["dueling/q_pred_std"] = float(q.std(unbiased=False).cpu().item()) if q.numel() > 1 else 0.0
            metrics["dueling/q_pred_min"] = float(q.min().cpu().item())
            metrics["dueling/q_pred_max"] = float(q.max().cpu().item())

        advantages = outputs.get("advantages")
        if isinstance(advantages, torch.Tensor) and advantages.numel() > 0:
            a = advantages.detach().float()
            metrics["dueling/advantage_std"] = (
                float(a.std(unbiased=False).cpu().item()) if a.numel() > 1 else 0.0
            )
            metrics["dueling/advantage_min"] = float(a.min().cpu().item())
            metrics["dueling/advantage_max"] = float(a.max().cpu().item())

        centered_advantages = outputs.get("centered_advantages")
        if isinstance(centered_advantages, torch.Tensor) and centered_advantages.numel() > 0:
            ca = centered_advantages.detach().float()
            metrics["dueling/centered_advantage_std"] = (
                float(ca.std(unbiased=False).cpu().item()) if ca.numel() > 1 else 0.0
            )

        return metrics

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Merge latest dueling diagnostics into normal Trainer logs.

        HuggingFace Trainer calls log() according to logging_steps. By merging
        here, we preserve the standard logging cadence while adding q_loss,
        value_loss, q-value statistics, etc.
        """
        dueling_metrics = getattr(self, "_last_dueling_metrics", None)
        if dueling_metrics:
            # Do not overwrite built-in keys such as "loss" or "learning_rate".
            for key, value in dueling_metrics.items():
                logs.setdefault(key, value)

        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Save DuelingQNet checkpoints in a normal loadable format.

        This path supports normal multi-GPU DDP. It intentionally rejects FSDP, because FSDP previously produced non-loadable rank-local/flattened tensors.

        Saved files:
          - pytorch_model.bin
          - config.json
          - dueling_qnet_config.json
          - tokenizer files
          - training_args.bin
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_fsdp_enabled:
            raise RuntimeError(
                "FSDP saving is disabled for DuelingQNet. "
                "Use multi-GPU DDP via torchrun, but remove --fsdp."
            )

        # In DDP this unwraps DistributedDataParallel -> DuelingQNet.
        model = self.accelerator.unwrap_model(self.model)

        # Important: use the unwrapped model state_dict, not a sharded/flattened one.
        if state_dict is None:
            state_dict = model.state_dict()

        if self.args.should_save:
            model.save_pretrained(
                output_dir,
                is_main_process=True,
                save_function=torch.save,
                state_dict=state_dict,
            )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Evaluation step.

        For the first implementation it returned only eval loss. Candidate sets have variable K, so returning q_values/labels to the default HF gather
        logic can easily fail when it tries to concatenate vectors of different lengths. A richer custom eval loop can be added later.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
                if not isinstance(outputs, dict) or "loss" not in outputs:
                    raise ValueError("DuelingQNet must return a dict containing 'loss' during evaluation.")
                loss = outputs["loss"].mean().detach()
                self._last_dueling_metrics = self._extract_dueling_metrics(outputs)

        return loss, None, None
