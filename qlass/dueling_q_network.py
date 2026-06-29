"""Dueling Q-network for candidate-normalized parametric-action critic.

This module implements the model side of the grouped dueling dataset produced by qlass.dueling_data:
    value_input_ids / value_attention_mask: one state-only prompt, shape [1, T]
    adv_input_ids / adv_attention_mask: K state+action prompts, shape [K, T]
    candidate_targets: optional targets for K candidates, shape [K]

The architecture follows the variant B discussed for QLASS:
    shared Llama backbone
      - V-head reads a state-only representation and predicts V(s)
      - A-head reads state-action representations and predicts A(s, a_i)

Candidate-normalized Q values are then
    Q_i = V(s) + A(s, a_i) - mean_j A(s, a_j)

For the initial implementation I intentionally support the batch-size-1 grouped collator only: one parent state and its K candidates per training batch.
This keeps the dueling normalization exact and avoids padding/masking over variable K.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class DuelingQNetOutput(dict):
    """Small dict subclass so callers can use both output["loss"] and attributes."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _getattr_or_default(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _resolve_model_dtype(training_args: Any = None) -> torch.dtype:
    """Resolve the dtype used to load the backbone.

    QLASS q-net training is normally run with ``--bf16 True``.  Keep this
    helper defensive so the model can also be instantiated in CPU/unit tests.
    """
    if bool(_getattr_or_default(training_args, "bf16", False)):
        return torch.bfloat16
    if bool(_getattr_or_default(training_args, "fp16", False)):
        return torch.float16
    return torch.float32


def _make_scalar_head(hidden_size: int, head_hidden_size: int = 1024) -> nn.Sequential:
    """MLP scalar head mirroring the existing QNet head style."""
    return nn.Sequential(
        nn.Linear(hidden_size, head_hidden_size),
        nn.ReLU(),
        nn.Linear(head_hidden_size, head_hidden_size),
        nn.ReLU(),
        nn.Linear(head_hidden_size, 1, bias=False),
    )


class DuelingQNet(nn.Module):
    """Shared-backbone dueling critic for generated candidate actions.

    Parameters are intentionally close to the existing QNet constructor so
    the class can be plugged into a new train script with minimal friction.
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        args: Any = None,
        *,
        pad_token_id: Optional[int] = None,
        no_load: bool = False,
        model_args: Any = None,
        training_args: Any = None,
        value_loss_coef: Optional[float] = None,
        head_hidden_size: int = 1024,
    ) -> None:
        super().__init__()

        # Keep compatibility with the old QNet call pattern where ``args`` and
        # ``model_args`` are sometimes the same object.
        if model_args is None:
            model_args = args
        if model_args is None:
            raise ValueError("DuelingQNet requires model_args or args with model_name_or_path.")

        model_name_or_path = model_args.model_name_or_path
        trust_remote_code = bool(_getattr_or_default(model_args, "trust_remote_code", False))
        cache_dir = _getattr_or_default(training_args, "cache_dir", None)
        torch_dtype = _resolve_model_dtype(training_args)

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        config.use_cache = False

        if pad_token_id is not None:
            config.pad_token_id = int(pad_token_id)

        # ``no_load=True`` is used by from_pretrained(): instantiate from config
        # first, then load the full DuelingQNet state_dict with llama/value/adv keys.
        if no_load:
            self.llama = AutoModelForCausalLM.from_config(config)
            if torch_dtype != torch.float32:
                self.llama = self.llama.to(dtype=torch_dtype)
        else:
            from_pretrained_kwargs: Dict[str, Any] = {
                "config": config,
                "cache_dir": cache_dir,
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
            }

            # Match the current QNet behavior: use FlashAttention-2 when the
            # environment supports it.  I enable it by default for the eto-style
            # Q training path because current QNet already does that.
            use_flash_attn = bool(_getattr_or_default(model_args, "use_flash_attn", True))
            if use_flash_attn:
                from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"

            self.llama = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **from_pretrained_kwargs,
            )

        self.config = self.llama.config
        if pad_token_id is not None:
            self.config.pad_token_id = int(pad_token_id)

        if hidden_size is None:
            hidden_size = int(self.config.hidden_size)
        self.hidden_size = int(hidden_size)
        self.head_hidden_size = int(head_hidden_size)

        backbone_dtype = next(self.llama.parameters()).dtype
        self.value_head = _make_scalar_head(self.hidden_size, self.head_hidden_size).to(dtype=backbone_dtype)
        self.advantage_head = _make_scalar_head(self.hidden_size, self.head_hidden_size).to(dtype=backbone_dtype)

        # In linear dueling decomposition sigmoid should NOT be applied to V/A.
        # Applying sigmoid to V or A breaks Q = V + centered(A).
        self.value_loss_coef = float(
            value_loss_coef
            if value_loss_coef is not None
            else _getattr_or_default(model_args, "dueling_value_loss_coef", 0.1)
        )

        self.dueling_config = {
            "architectural_type": "candidate_normalized_parametric_action_dueling_critic",
            "base_model_name_or_path": model_name_or_path,
            "hidden_size": self.hidden_size,
            "head_hidden_size": self.head_hidden_size,
            "value_loss_coef": self.value_loss_coef,
        }

    @staticmethod
    def _last_non_pad_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the hidden state at the last non-padding token.

        This is robust to both right and left padding.  The existing QNet uses a pad-token based sequence length heuristic;
        using attention_mask directly is safer.
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected hidden_states with shape [B, T, H], got {hidden_states.shape}")
        if attention_mask.dim() != 2:
            raise ValueError(f"Expected attention_mask with shape [B, T], got {attention_mask.shape}")
        if hidden_states.shape[:2] != attention_mask.shape:
            raise ValueError(
                "hidden_states and attention_mask shape mismatch: "
                f"hidden_states={hidden_states.shape}, attention_mask={attention_mask.shape}"
            )

        batch_size, seq_len, _ = hidden_states.shape
        mask = attention_mask.to(device=hidden_states.device, dtype=torch.long)
        if torch.any(mask.sum(dim=1) <= 0):
            raise ValueError("Each sequence must contain at least one non-padding token.")

        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        last_indices = positions.masked_fill(mask == 0, -1).max(dim=1).values
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), last_indices]

    def _encode_and_pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]
        return self._last_non_pad_pool(last_hidden, attention_mask)

    def _encode_value_and_advantage(
        self,
        value_input_ids: torch.Tensor,
        value_attention_mask: torch.Tensor,
        adv_input_ids: torch.Tensor,
        adv_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode state-only and state-action inputs with the shared backbone.

        If both tensors have the same sequence length, concatenate them and make one backbone call.
        This is faster and still exactly the same shared encoder. If dynamic padding gives different lengths, fall back to two backbone calls.
        """
        if value_input_ids.size(1) == adv_input_ids.size(1):
            all_input_ids = torch.cat([value_input_ids, adv_input_ids], dim=0)
            all_attention_mask = torch.cat([value_attention_mask, adv_attention_mask], dim=0)
            pooled = self._encode_and_pool(all_input_ids, all_attention_mask)
            value_pooled = pooled[: value_input_ids.size(0)]
            adv_pooled = pooled[value_input_ids.size(0) :]
            return value_pooled, adv_pooled

        value_pooled = self._encode_and_pool(value_input_ids, value_attention_mask)
        adv_pooled = self._encode_and_pool(adv_input_ids, adv_attention_mask)
        return value_pooled, adv_pooled

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean()
        mask_f = mask.to(device=values.device, dtype=values.dtype)
        denom = mask_f.sum().clamp_min(1.0)
        return (values * mask_f).sum() / denom

    def forward(
        self,
        *,
        value_input_ids: torch.Tensor,
        value_attention_mask: torch.Tensor,
        adv_input_ids: torch.Tensor,
        adv_attention_mask: torch.Tensor,
        candidate_targets: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        num_candidates: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> DuelingQNetOutput | tuple:
        """Forward pass for one grouped state and K candidates.

        Current supported shapes from DuelingDataCollatorForBatchSizeOne:

            value_input_ids:      [1, T]
            value_attention_mask: [1, T]
            adv_input_ids:        [K, T]
            adv_attention_mask:   [K, T]
            candidate_targets:    [K], optional
            candidate_mask:       [K], optional; all True for now
        """
        if value_input_ids.dim() != 2 or value_input_ids.size(0) != 1:
            raise ValueError(
                "DuelingQNet currently expects one state per batch: "
                f"value_input_ids shape must be [1, T], got {tuple(value_input_ids.shape)}."
            )
        if adv_input_ids.dim() != 2:
            raise ValueError(f"adv_input_ids must have shape [K, T], got {tuple(adv_input_ids.shape)}")
        if adv_input_ids.size(0) < 2:
            raise ValueError(f"Dueling training needs at least 2 candidates, got K={adv_input_ids.size(0)}")

        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(device=adv_input_ids.device, dtype=torch.bool).view(-1)
            if candidate_mask.numel() != adv_input_ids.size(0):
                raise ValueError(
                    f"candidate_mask length {candidate_mask.numel()} != num candidates {adv_input_ids.size(0)}"
                )
        else:
            candidate_mask = torch.ones(adv_input_ids.size(0), device=adv_input_ids.device, dtype=torch.bool)

        value_pooled, adv_pooled = self._encode_value_and_advantage(
            value_input_ids=value_input_ids,
            value_attention_mask=value_attention_mask,
            adv_input_ids=adv_input_ids,
            adv_attention_mask=adv_attention_mask,
        )

        # [1, 1] -> scalar; [K, 1] -> [K]
        value = self.value_head(value_pooled).view(-1)[0]
        advantages = self.advantage_head(adv_pooled).view(-1)

        advantage_mean = self._masked_mean(advantages, candidate_mask)
        centered_advantages = advantages - advantage_mean
        q_values = value + centered_advantages

        # Cast predictions used in loss/logging to fp32 for numerical stability.
        value_fp32 = value.float()
        advantages_fp32 = advantages.float()
        advantage_mean_fp32 = advantage_mean.float()
        centered_advantages_fp32 = centered_advantages.float()
        q_values_fp32 = q_values.float()

        output = DuelingQNetOutput(
            value=value_fp32,                              # scalar tensor
            advantages=advantages_fp32,                    # [K]
            advantage_mean=advantage_mean_fp32,            # scalar tensor
            centered_advantages=centered_advantages_fp32,  # [K]
            q_values=q_values_fp32,                        # [K]
        )

        if candidate_targets is not None:
            targets = candidate_targets.to(device=q_values_fp32.device, dtype=torch.float32).view(-1)
            if targets.numel() != q_values_fp32.numel():
                raise ValueError(
                    f"candidate_targets length {targets.numel()} != q_values length {q_values_fp32.numel()}"
                )
            mask_f = candidate_mask.to(device=q_values_fp32.device, dtype=torch.float32)
            denom = mask_f.sum().clamp_min(1.0)

            q_loss = (((q_values_fp32 - targets) ** 2) * mask_f).sum() / denom

            # Since mean_i Q_i = V(s) under candidate-normalized dueling, the
            # natural auxiliary value target is mean_i target_i for this group.
            value_target = (targets * mask_f).sum() / denom
            value_loss = F.mse_loss(value_fp32, value_target.detach())
            loss = q_loss + self.value_loss_coef * value_loss

            output.update(
                loss=loss,
                q_loss=q_loss.detach(),
                value_loss=value_loss.detach(),
                value_target=value_target.detach(),
            )

        if return_dict:
            return output

        # Tuple order is intentionally simple; custom trainer should prefer dict.
        if "loss" in output:
            return output["loss"], output["q_values"], output["value"], output["advantages"]
        return output["q_values"], output["value"], output["advantages"]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        self.llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.llama.gradient_checkpointing_disable()

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        save_function: Any = torch.save,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Save full DuelingQNet state in the same spirit as the existing QNet.

        ``Trainer.save_model`` / FSDP may pass an already materialized full
        ``state_dict``.  If it is not provided, we use ``self.state_dict()``.
        """
        if not is_main_process:
            return

        os.makedirs(save_directory, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()

        save_function(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

        # Save the underlying Llama config so from_pretrained(no_load=True) can
        # instantiate the same architecture before loading the full state dict.
        self.config.save_pretrained(save_directory)

        with open(os.path.join(save_directory, "dueling_qnet_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.dueling_config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, load_directory: str, accelerator: Any = None, args: Any = None) -> "DuelingQNet":
        """Load a checkpoint saved by ``save_pretrained``.

        The method mirrors the existing QNet.from_pretrained pattern: construct a randomly initialized module from config,
        then load the full state_dict.
        """
        load_directory = str(load_directory)
        config = AutoConfig.from_pretrained(load_directory)

        # We need an args-like object with model_name_or_path for __init__.
        if args is None:
            class _Args:
                pass
            args = _Args()
        args.model_name_or_path = load_directory

        value_loss_coef = None
        dueling_config_path = Path(load_directory) / "dueling_qnet_config.json"
        if dueling_config_path.exists():
            with open(dueling_config_path, "r", encoding="utf-8") as f:
                dueling_config = json.load(f)
            value_loss_coef = float(dueling_config.get("value_loss_coef", 0.1))

        class _LoadTrainingArgs:
            # QLASS QNet checkpoints are normally trained/served in bf16.  Keep
            # bf16 as the default loading dtype unless the caller explicitly
            # provides args.bf16=False and args.fp16=True/False.
            cache_dir = None
            bf16 = bool(getattr(args, "bf16", True))
            fp16 = bool(getattr(args, "fp16", False))

        model = cls(
            hidden_size=int(config.hidden_size),
            args=args,
            no_load=True,
            model_args=args,
            training_args=_LoadTrainingArgs(),
            value_loss_coef=value_loss_coef,
        )

        state_dict_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")

        if accelerator is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

        return model
