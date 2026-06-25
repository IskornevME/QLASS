"""Dataset and collator for candidate-normalized parametric-action dueling critic.

The expected input file is produced by construct_dueling_q_data.py and has one JSON object per *state / parent node*:

    {
        "state_conversations": [...],          # common state s
        "candidates": [
            {"action": {...}, "target": 0.81, ...},
            {"action": {...}, "target": 0.00, ...},
        ],
        ...
    }

The collator below intentionally starts with the simplest and safest setting:
"per_device_train_batch_size = 1".  This keeps one candidate set intact in a
batch and avoids padding/masking bugs over variable numbers of candidates.
Gradient accumulation can still be used to recover a larger effective batch.
"""

import json
import math
import torch
import random
import transformers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import Dataset

from qlass.data_utils import get_chat_template, rank0_print


Conversation = List[Dict[str, str]]


def _to_messages(conversations: Conversation) -> List[Dict[str, Optional[str]]]:
    """Convert QLASS conversation format to ChatTemplate message format.

    QLASS stores messages as {"from": "human"|"gpt", "value": "..."}.
    ChatTemplate expects {"role": "user"|"assistant", "content": "..."}.

    This mirrors qlass.data_utils.preprocess(), including the behavior of skipping a leading non-human message if it ever appears.
    """
    roles = {"human": "user", "gpt": "assistant"}
    expected_roles = ["user", "assistant"]

    src = conversations
    if not src:
        raise ValueError("Empty conversation cannot be converted to messages.")

    if roles[src[0]["from"]] != "user":
        src = src[1:]

    messages: List[Dict[str, Optional[str]]] = []
    for j, msg in enumerate(src):
        role = roles[msg["from"]]
        if role != expected_roles[j % 2]:
            raise ValueError(
                "Conversation roles do not alternate after conversion: "
                f"position={j}, expected={expected_roles[j % 2]}, got={role}, "
                f"message={msg!r}"
            )
        messages.append({"role": role, "content": msg["value"]})

    return messages


def build_prompt_from_conversations(
    conversations: Conversation,
    *,
    model_path: str,
    add_generation_prompt: bool = False,
) -> str:
    """Build a model prompt using the same QLASS chat template.

    For advantage inputs, ``conversations`` should usually end with the
    candidate assistant action, and ``add_generation_prompt`` should be False.

    For value inputs, ``conversations`` usually ends with a human observation.
    ``add_generation_prompt=True`` appends an assistant prefix when the template
    has one.  For Llama-2-chat the assistant prefix is empty, so this is mostly
    a no-op, but keeping the flag makes the behavior correct for other chat
    templates such as Llama-3-Instruct.
    """
    chat = get_chat_template(model_path)
    messages = _to_messages(conversations)

    if add_generation_prompt:
        if messages and messages[-1]["role"] == "assistant":
            # State should normally end before the next action.  Do not append a
            # second assistant prefix if the user accidentally passes an action.
            prompt_messages = messages
        else:
            prompt_messages = messages + [{"role": "assistant", "content": None}]
    else:
        prompt_messages = messages

    return chat.get_prompt(prompt_messages)


def _target_std(targets: Sequence[float]) -> float:
    if len(targets) <= 1:
        return 0.0
    t = torch.tensor(targets, dtype=torch.float32)
    # Match numpy default std/ddof=0; this is what matters for zero-variance filtering.
    return float(torch.std(t, unbiased=False).item())


class DuelingGroupedDataset(Dataset):
    """Grouped dataset for candidate-normalized dueling Q training.

    Each item is one state with K candidate actions.  The dataset does not tokenize examples;
    tokenization is delegated to the collator so that memory usage stays small for large JSON files.

    Parameters
    ----------
    data_path:
        Path to ``dueling_vanilla_*.json``.
    target_key:
        Candidate field used as regression target. Usually ``"target"``.
    min_candidates:
        Keep only groups with at least this many candidates after any optional
        filtering/subsampling. For dueling training this should be >= 2.
    max_candidates:
        Optional cap on K. If a group has more than this many candidates, the
        candidates are either sampled or truncated depending on
        ``sample_candidates``.
    sample_candidates:
        If True and ``max_candidates`` is set, randomly sample candidates in
        ``__getitem__``. If False, keep the first ``max_candidates``.
    drop_zero_variance:
        If True, drop groups where all candidate targets are equal.  This can
        strengthen the advantage signal but may significantly shrink the data.
    drop_state_mismatch:
        If True, drop groups marked by the constructor as having children whose
        stored states are not identical.  Such groups are suspicious because
        dueling assumes candidates are alternatives from the same state.
    seed:
        seed used for candidate subsampling.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        target_key: str = "target",
        min_candidates: int = 2,
        max_candidates: Optional[int] = None,
        sample_candidates: bool = False,
        drop_zero_variance: bool = False,
        drop_state_mismatch: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_path = str(data_path)
        self.target_key = target_key
        self.min_candidates = int(min_candidates)
        self.max_candidates = max_candidates
        self.sample_candidates = bool(sample_candidates)
        self.drop_zero_variance = bool(drop_zero_variance)
        self.drop_state_mismatch = bool(drop_state_mismatch)
        self.rng = random.Random(seed)

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(
                f"Expected {self.data_path} to contain a JSON list, got {type(raw_data)}"
            )

        self.examples: List[Dict[str, Any]] = []
        counters = {
            "raw_groups": 0,
            "kept_groups": 0,
            "drop_missing_fields": 0,
            "drop_too_few_candidates": 0,
            "drop_zero_variance": 0,
            "drop_state_mismatch": 0,
        }

        for ex in raw_data:
            counters["raw_groups"] += 1

            if "state_conversations" not in ex or "candidates" not in ex:
                counters["drop_missing_fields"] += 1
                continue

            candidates = ex["candidates"]
            if not isinstance(candidates, list):
                counters["drop_missing_fields"] += 1
                continue

            valid_candidates = []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                if "action" not in cand or self.target_key not in cand:
                    continue
                try:
                    target = float(cand[self.target_key])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(target):
                    continue

                # Store a normalized float copy under a stable key used by the collator.
                cand_copy = dict(cand)
                cand_copy["_dueling_target"] = target
                valid_candidates.append(cand_copy)

            if len(valid_candidates) < self.min_candidates:
                counters["drop_too_few_candidates"] += 1
                continue

            if self.drop_state_mismatch and bool(ex.get("state_mismatch_across_children", False)):
                counters["drop_state_mismatch"] += 1
                continue

            targets = [cand["_dueling_target"] for cand in valid_candidates]
            if self.drop_zero_variance and _target_std(targets) <= 0.0:
                counters["drop_zero_variance"] += 1
                continue

            ex_copy = dict(ex)
            ex_copy["candidates"] = valid_candidates
            self.examples.append(ex_copy)
            counters["kept_groups"] += 1

        rank0_print(
            "[DuelingGroupedDataset] " + " ".join(f"{k}={v}" for k, v in counters.items())
        )

        if len(self.examples) == 0:
            raise ValueError(
                "DuelingGroupedDataset is empty after filtering. "
                f"data_path={self.data_path}, min_candidates={self.min_candidates}, "
                f"drop_zero_variance={self.drop_zero_variance}, "
                f"drop_state_mismatch={self.drop_state_mismatch}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def _select_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.max_candidates is None or len(candidates) <= self.max_candidates:
            return candidates

        if self.sample_candidates:
            # Sampling happens per __getitem__; this is useful if future MCTS runs produce larger K and we want candidate dropout.
            return self.rng.sample(candidates, self.max_candidates)

        return candidates[: self.max_candidates]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        candidates = self._select_candidates(ex["candidates"])

        if len(candidates) < self.min_candidates:
            # This can only happen if max_candidates < min_candidates.
            raise RuntimeError(
                f"Selected too few candidates: {len(candidates)} < {self.min_candidates}"
            )

        return {
            "id": ex.get("id"),
            "parent_node_id": ex.get("parent_node_id"),
            "parent_depth": ex.get("parent_depth"),
            "state_conversations": ex["state_conversations"],
            "candidates": candidates,
            "state_mismatch_across_children": bool(ex.get("state_mismatch_across_children", False)),
            "num_candidates": len(candidates),
        }


@dataclass
class DuelingDataCollatorForBatchSizeOne:
    """Tokenize one grouped state and its K candidate actions.

    The output shapes are:

        value_input_ids:        [1, T]
        value_attention_mask:   [1, T]
        adv_input_ids:          [K, T]
        adv_attention_mask:     [K, T]
        candidate_targets:      [K]
        candidate_mask:         [K]  # all True in this batch-size-1 version

    This collator intentionally rejects ``len(features) != 1``. Variable-K batching requires either padding over K 
    or flattened segment aggregation in the trainer/model. Starting with batch size 1 avoids that complexity.
    """

    tokenizer: transformers.PreTrainedTokenizer
    model_path: str
    model_max_length: int = 4096
    pad_to_max_length: bool = True
    add_generation_prompt_to_value: bool = True
    return_metadata: bool = False

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer must have pad_token_id set before using "
                "DuelingDataCollatorForBatchSizeOne."
            )

    def _tokenize_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        padding = "max_length" if self.pad_to_max_length else True

        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=padding,
            max_length=self.model_max_length,
            truncation=True,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "DuelingDataCollatorForBatchSizeOne requires per_device_train_batch_size=1. "
                f"Got batch size {len(features)}."
            )

        ex = features[0]
        state_conversations: Conversation = ex["state_conversations"]
        candidates: List[Dict[str, Any]] = ex["candidates"]

        if len(candidates) < 2:
            raise ValueError(
                "Dueling collator received a group with fewer than 2 candidates. "
                "Check dataset filtering."
            )

        value_prompt = build_prompt_from_conversations(
            state_conversations,
            model_path=self.model_path,
            add_generation_prompt=self.add_generation_prompt_to_value,
        )

        adv_prompts: List[str] = []
        targets: List[float] = []
        action_lines: List[str] = []

        for cand in candidates:
            # Advantage input is exactly "state + candidate assistant action".
            adv_conversation = state_conversations + [cand["action"]]
            adv_prompts.append(
                build_prompt_from_conversations(
                    adv_conversation,
                    model_path=self.model_path,
                    add_generation_prompt=False,
                )
            )
            targets.append(float(cand["_dueling_target"]))
            action_lines.append(str(cand.get("action_line") or cand.get("normalized_action") or ""))

        value_tokens = self._tokenize_prompts([value_prompt])
        adv_tokens = self._tokenize_prompts(adv_prompts)

        batch: Dict[str, Any] = {
            "value_input_ids": value_tokens["input_ids"],
            "value_attention_mask": value_tokens["attention_mask"],
            "adv_input_ids": adv_tokens["input_ids"],
            "adv_attention_mask": adv_tokens["attention_mask"],
            "candidate_targets": torch.tensor(targets, dtype=torch.float32),
            "candidate_mask": torch.ones(len(targets), dtype=torch.bool),
            "num_candidates": torch.tensor([len(targets)], dtype=torch.long),
        }

        if self.return_metadata:
            # Keep metadata optional: HuggingFace Trainer can move nested tensors
            # to device, but arbitrary strings/lists should not be passed to the
            # model unless the custom trainer explicitly removes them.
            batch["metadata"] = {
                "id": ex.get("id"),
                "parent_node_id": ex.get("parent_node_id"),
                "parent_depth": ex.get("parent_depth"),
                "state_mismatch_across_children": ex.get("state_mismatch_across_children"),
                "action_lines": action_lines,
            }

        return batch


def make_dueling_data_module(
    *,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    model_max_length: int = 4096,
    min_candidates: int = 2,
    max_candidates: Optional[int] = None,
    sample_candidates: bool = False,
    drop_zero_variance: bool = False,
    drop_state_mismatch: bool = False,
    pad_to_max_length: bool = True,
) -> Dict[str, Any]:
    """Small helper mirroring the style of ``make_supervised_data_module``."""
    train_dataset = DuelingGroupedDataset(
        train_data_path,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        sample_candidates=sample_candidates,
        drop_zero_variance=drop_zero_variance,
        drop_state_mismatch=drop_state_mismatch,
    )

    eval_dataset = None
    if eval_data_path:
        eval_dataset = DuelingGroupedDataset(
            eval_data_path,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            sample_candidates=False,
            drop_zero_variance=drop_zero_variance,
            drop_state_mismatch=drop_state_mismatch,
        )

    data_collator = DuelingDataCollatorForBatchSizeOne(
        tokenizer=tokenizer,
        model_path=model_path,
        model_max_length=model_max_length,
        pad_to_max_length=pad_to_max_length,
        return_metadata=False,
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
