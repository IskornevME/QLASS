"""Train a candidate-normalized parametric-action dueling QNet.

This is the dueling counterpart of ``train_q.py``.  It uses:
  - ``DuelingGroupedDataset`` / ``DuelingDataCollatorForBatchSizeOne``;
  - ``DuelingQNet``;
  - ``DuelingQTrainer``.

The first implementation intentionally keeps one parent candidate group per batch. Use gradient accumulation to get a larger effective batch size.
"""
import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
sys.path.append(str(Path(ROOT_DIR).parent))

from qlass.data_utils import rank0_print
from qlass.dueling_data import make_dueling_data_module
from qlass.dueling_q_network import DuelingQNet
from qlass.dueling_q_trainer import DuelingQTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/home/m.iskornev/qlass/models/qlass-Llama-2-7b-chat-hf-alfworld-sft_run1",
        metadata={"help": "Base/SFT Llama model path used as shared dueling backbone."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to allow custom model code from the model repository."},
    )
    padding_side: str = field(default="right", metadata={"help": "Tokenizer padding side."})
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use FlashAttention-2 when loading the backbone if supported."},
    )
    dueling_value_loss_coef: float = field(
        default=0.1,
        metadata={"help": "Coefficient for auxiliary V(s) regression to mean candidate target."},
    )
    dueling_head_hidden_size: int = field(
        default=1024,
        metadata={"help": "Hidden size of value/advantage MLP heads."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/m.iskornev/qlass/QLASS/data/train/alfworld/explore_7b_sft_d8_i0_s2_mpr3/dueling_vanilla_raw.json",
        metadata={"help": "Path to dueling_vanilla_*.json training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to dueling_vanilla_*.json evaluation data."},
    )
    min_candidates: int = field(
        default=2,
        metadata={"help": "Keep groups with at least this many candidates."},
    )
    max_candidates: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on candidate count K within one group."},
    )
    sample_candidates: bool = field(
        default=False,
        metadata={"help": "Randomly sample candidates when max_candidates is set."},
    )
    drop_zero_variance: bool = field(
        default=False,
        metadata={"help": "Drop groups where all candidate targets are equal."},
    )
    drop_state_mismatch: bool = field(
        default=False,
        metadata={"help": "Drop groups whose sibling child states differ."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Pad value/advantage prompts to model_max_length in the collator."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum tokenized prompt length."},
    )


def trainer_save_model_safe(trainer: transformers.Trainer) -> None:
    """Save model safely in single-GPU and FSDP settings.

    This mirrors the helper used by ``train_q.py``. ``DuelingQNet`` implements ``save_pretrained``, so ``trainer.save_model`` will call it correctly.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        if isinstance(trainer.model, FSDP):
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
                trainer.save_model()
        else:
            trainer.save_model()
    except Exception:
        trainer.save_model()


def _validate_training_setup(training_args: TrainingArguments) -> None:
    """Fail early for settings incompatible with the first dueling collator."""
    if training_args.per_device_train_batch_size != 1:
        raise ValueError(
            "DuelingDataCollatorForBatchSizeOne requires --per_device_train_batch_size 1. "
            f"Got {training_args.per_device_train_batch_size}. Use --gradient_accumulation_steps "
            "to increase the effective batch size."
        )

    if training_args.do_eval and training_args.per_device_eval_batch_size != 1:
        raise ValueError(
            "The initial dueling eval path also expects --per_device_eval_batch_size 1. "
            f"Got {training_args.per_device_eval_batch_size}."
        )

    if training_args.remove_unused_columns:
        # The custom collator emits keys such as value_input_ids/adv_input_ids.
        # Keeping unused-column removal enabled can silently remove needed fields when Trainer wraps the collator.
        rank0_print(
            "[train_dueling_q] Forcing remove_unused_columns=False for dueling batches."
        )
        training_args.remove_unused_columns = False


def train() -> None:
    torch.set_default_dtype(torch.float32)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["RANK"] = str(training_args.local_rank)
    _validate_training_setup(training_args)

    # Load config once here for hidden_size and optional RoPE warning. The actual backbone is loaded inside DuelingQNet to keep the model self-contained.
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        rank0_print(
            "[train_dueling_q] WARNING: model_max_length exceeds base max_position_embeddings: "
            f"model_max_length={training_args.model_max_length}, orig_ctx_len={orig_ctx_len}, "
            f"suggested linear rope factor={scaling_factor}. Current DuelingQNet loads its own config; "
            "use <= original context length unless you explicitly add RoPE scaling to DuelingQNet."
        )
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have pad_token_id after setup.")

    model = DuelingQNet(
        hidden_size=config.hidden_size,
        args=model_args,
        pad_token_id=tokenizer.pad_token_id,
        model_args=model_args,
        training_args=training_args,
        value_loss_coef=model_args.dueling_value_loss_coef,
        head_hidden_size=model_args.dueling_head_hidden_size,
    )

    data_module = make_dueling_data_module(
        tokenizer=tokenizer,
        model_path=model_args.model_name_or_path,
        train_data_path=data_args.data_path,
        eval_data_path=data_args.eval_data_path,
        model_max_length=training_args.model_max_length,
        min_candidates=data_args.min_candidates,
        max_candidates=data_args.max_candidates,
        sample_candidates=data_args.sample_candidates,
        drop_zero_variance=data_args.drop_zero_variance,
        drop_state_mismatch=data_args.drop_state_mismatch,
        pad_to_max_length=data_args.pad_to_max_length,
    )

    trainer = DuelingQTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    checkpoint_exists = bool(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")))
    if checkpoint_exists:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
