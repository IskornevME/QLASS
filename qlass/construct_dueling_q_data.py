"""
Build grouped Q-data for a candidate-normalized dueling critic.

The original QLASS Q-data flattens every tree node into one example:
    conversations = node.state + [node.action]
    label         = node.q_value

For a dueling critic we must preserve sibling candidate sets. One training
example is therefore a *decision state* plus all actions expanded from that state:
    state_conversations = common state before all child actions
    candidates          = [{action, target}, ...]

Important TreeNode convention in QLASS:
    child.state is the state BEFORE child.action.
So for a candidate set represented by parent.children, the shared state should be inferred from the children, not from parent.state.
root.state, for example, can be a plain string, while root.children[*].state contains the actual chat history used before the first action.
"""

import argparse
import copy
import json
import math
import os
import pickle
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

# Make the script runnable both as `python qlass/construct_dueling_q_data.py`
# and as `python construct_dueling_q_data.py` from the repository root.
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
sys.path.append(str(Path(ROOT_DIR).parent))

# Importing TreeNode is needed for pickle deserialization of old exploration trees.
# The script does not rely on isinstance(node, TreeNode), because older pickles can still be loaded as long as objects expose the same attributes.
from qlass.explore_sft_agent import TreeNode


# -----------------------------
# Basic serialization utilities
# -----------------------------


def to_float_or_none(value: Any) -> Optional[float]:
    """Convert numeric/bool rewards and q-values to JSON-friendly floats."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def action_to_text(action: Any) -> str:
    """Return assistant action text from a TreeNode.action-like object."""
    if isinstance(action, dict):
        return str(action.get("value", ""))
    return str(action)


def action_to_message(action: Any) -> Dict[str, str]:
    """Convert action to a canonical QLASS conversation message."""
    if isinstance(action, dict):
        return {
            "from": str(action.get("from", "gpt")),
            "value": str(action.get("value", "")),
        }
    return {"from": "gpt", "value": str(action)}


def normalize_message(msg: Dict[str, Any]) -> Dict[str, str]:
    """Normalize both {from,value} and {role,content} chat message formats."""
    if "from" in msg and "value" in msg:
        role = str(msg["from"])
        value = str(msg["value"])
    elif "role" in msg and "content" in msg:
        role_map = {"user": "human", "assistant": "gpt", "system": "human"}
        role = role_map.get(str(msg["role"]), str(msg["role"]))
        value = "" if msg["content"] is None else str(msg["content"])
    else:
        raise ValueError(f"Unsupported message format: {msg!r}")

    if role == "assistant":
        role = "gpt"
    if role == "user":
        role = "human"
    return {"from": role, "value": value}


def normalize_conversations(state: Any) -> List[Dict[str, str]]:
    """Convert a TreeNode.state-like object to a list of QLASS messages.

    Most non-root nodes already store a list of {from,value} messages. Some root nodes store only the initial observation/task as a string; this helper
    makes that case explicit and JSON-serializable.
    """
    if state is None:
        return []
    if isinstance(state, str):
        return [{"from": "human", "value": state}]
    if isinstance(state, list):
        out = []
        for msg in state:
            if isinstance(msg, dict):
                out.append(normalize_message(msg))
            else:
                out.append({"from": "human", "value": str(msg)})
        return out
    if isinstance(state, dict):
        return [normalize_message(state)]
    return [{"from": "human", "value": str(state)}]


def state_signature(state_conversations: List[Dict[str, str]]) -> str:
    """Stable signature for comparing sibling states."""
    return json.dumps(state_conversations, ensure_ascii=False, sort_keys=True)


# -----------------------------
# Action-line deduplication
# -----------------------------

_ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.*?)\s*$", flags=re.IGNORECASE | re.MULTILINE)


def extract_action_line(action_text: str) -> str:
    """Extract the last `Action: ...` line from a Thought/Action response.

    If the model output does not contain an explicit Action line, the whole text
    is used as a fallback. This makes the function robust to malformed outputs.
    """
    matches = list(_ACTION_RE.finditer(action_text or ""))
    if matches:
        line = matches[-1].group(1).strip()
        if line:
            return line
    return (action_text or "").strip()


def normalize_action_line(action: Any) -> Tuple[str, str]:
    """Return (raw_action_line, normalized_action_line)."""
    text = action_to_text(action)
    raw_line = extract_action_line(text)
    normalized = " ".join(raw_line.lower().split())
    return raw_line, normalized


# -----------------------------
# Tree loading / combining
# -----------------------------


def load_pickle_stream(file_path: str) -> List[Any]:
    """Load all pickle objects from a file.

    Exploration files usually contain a single pickled list, but the original QLASS helper also supports a stream of multiple pickle.dump calls. 
    I keep that behavior for compatibility.
    """
    objects = []
    with open(file_path, "rb") as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects


def combine_pkl_files(input_files: List[str], output_file: str) -> None:
    """Combine exploration tree pickle files into one combined_tree.pkl."""
    combined = []
    for file_path in input_files:
        for obj in load_pickle_stream(file_path):
            combined.append(obj)
    with open(output_file, "wb") as f:
        pickle.dump(combined, f)


def iter_tree_records(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield {'id': ..., 'tree': TreeNode} records from nested pickle payloads."""
    if isinstance(obj, dict) and "tree" in obj:
        yield obj
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            yield from iter_tree_records(item)


def load_tree_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Load tree records either from --combined_tree_path or from --data_path."""
    if args.combined_tree_path:
        combined_tree_path = args.combined_tree_path
    else:
        if not args.data_path:
            raise ValueError("Either --combined_tree_path or --data_path must be provided.")
        data_path = Path(args.data_path)
        combined_tree_path = str(data_path / "combined_tree.pkl")
        if not Path(combined_tree_path).exists() or args.rebuild_combined_tree:
            pkl_files = sorted(
                str(p)
                for p in data_path.iterdir()
                if p.suffix == ".pkl" and p.name != "combined_tree.pkl"
            )
            if not pkl_files:
                raise FileNotFoundError(f"No .pkl files found in {data_path}")
            print(f"[combine] Combining {len(pkl_files)} pkl files -> {combined_tree_path}")
            combine_pkl_files(pkl_files, combined_tree_path)

    print(f"[load] Loading trees from {combined_tree_path}")
    payload_objects = load_pickle_stream(combined_tree_path)
    records: List[Dict[str, Any]] = []
    for obj in payload_objects:
        records.extend(iter_tree_records(obj))

    if not records:
        raise RuntimeError(
            f"No tree records found in {combined_tree_path}. Expected nested dicts with keys 'id' and 'tree'."
        )
    return records


# -----------------------------
# Q-target preparation
# -----------------------------


def update_vanilla_rewards(node: Any) -> None:
    """Match the vanilla QLASS target setup: intermediate rewards are zero.

    Leaves keep their terminal reward; non-leaf nodes get reward=0 and then q_value is computed by TreeNode.update_q_values(gamma).
    """
    children = getattr(node, "children", []) or []
    if children:
        node.reward = 0
        for child in children:
            update_vanilla_rewards(child)


def minmax_normalize(values: List[float]) -> Tuple[List[float], float, float]:
    """Min-max normalize values; if degenerate, return zeros like vanilla code."""
    if not values:
        return [], 0.0, 0.0
    min_val = min(values)
    max_val = max(values)
    if max_val <= min_val:
        return [0.0 for _ in values], float(min_val), float(max_val)
    return [(v - min_val) / (max_val - min_val) for v in values], float(min_val), float(max_val)


# -----------------------------
# Optional group-consistent trimming
# -----------------------------


def maybe_init_tokenizer(args: argparse.Namespace):
    """Initialize tokenizer/chat template only when trimming is requested."""
    if not args.model_name_or_path:
        return None, None

    import transformers
    from qlass.data_utils import get_chat_template

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    chat = get_chat_template(args.model_name_or_path)
    return tokenizer, chat


def to_chat_messages(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert QLASS {from, value} messages to chat-template {role,content}."""
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    messages = []
    for msg in conversations:
        role = roles.get(msg["from"], msg["from"])
        messages.append({"role": role, "content": msg["value"]})

    # QLASS preprocess skips a leading assistant message.  Keep the same behavior.
    if messages and messages[0]["role"] != "user":
        messages = messages[1:]
    return messages


def prompt_num_tokens(conversations: List[Dict[str, str]], tokenizer: Any, chat: Any) -> int:
    messages = to_chat_messages(conversations)
    prompt = chat.get_prompt(messages)
    return len(tokenizer(prompt, add_special_tokens=True).input_ids)


def trim_state_group_consistently(
    state_conversations: List[Dict[str, str]],
    candidate_actions: List[Dict[str, str]],
    tokenizer: Any,
    chat: Any,
    max_prompt_tokens: int,
    keep_first_n: int,
    min_tail_msgs: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Trim the shared state once so every state+action_i sees the same state.

    We measure the maximum token length over all state+candidate_action prompts.
    Then we remove the oldest turns after the prefix, always by two messages, to preserve user/assistant alternation.
    """
    state = copy.deepcopy(state_conversations)

    def max_len_for_state(cur_state: List[Dict[str, str]]) -> int:
        if not candidate_actions:
            return prompt_num_tokens(cur_state, tokenizer, chat)
        return max(prompt_num_tokens(cur_state + [act], tokenizer, chat) for act in candidate_actions)

    before_tokens = max_len_for_state(state)
    trimmed = False

    if before_tokens <= max_prompt_tokens:
        return state, {
            "trimmed": False,
            "tokens_before": before_tokens,
            "tokens_after": before_tokens,
            "overflow_after_trim": False,
            "num_state_messages_before": len(state_conversations),
            "num_state_messages_after": len(state),
        }

    keep_n = min(keep_first_n, len(state))
    prefix = state[:keep_n]
    rest = state[keep_n:]
    cur_tokens = before_tokens

    while cur_tokens > max_prompt_tokens and len(rest) > min_tail_msgs:
        if len(rest) >= 2:
            rest = rest[2:]
        else:
            break
        state = prefix + rest
        cur_tokens = max_len_for_state(state)
        trimmed = True

    return state, {
        "trimmed": trimmed,
        "tokens_before": before_tokens,
        "tokens_after": cur_tokens,
        "overflow_after_trim": cur_tokens > max_prompt_tokens,
        "num_state_messages_before": len(state_conversations),
        "num_state_messages_after": len(state),
    }


# -----------------------------
# Group collection
# -----------------------------


def build_candidate_from_child(child: Any) -> Optional[Dict[str, Any]]:
    """Create a JSON-serializable candidate record from a child TreeNode."""
    if getattr(child, "action", None) is None:
        return None
    raw_target = to_float_or_none(getattr(child, "q_value", None))
    if raw_target is None:
        return None

    action_msg = action_to_message(child.action)
    action_line, normalized_action = normalize_action_line(child.action)

    return {
        "child_node_id": int(getattr(child, "id", -1)),
        "action": action_msg,
        "action_text": action_msg["value"],
        "action_line": action_line,
        "normalized_action": normalized_action,
        "raw_target": float(raw_target),
        # Filled after optional per-tree normalization.
        "target": None,
        "reward": to_float_or_none(getattr(child, "reward", None)),
        "is_terminal": len(getattr(child, "children", []) or []) == 0,
        "num_child_children": len(getattr(child, "children", []) or []),
    }


def deduplicate_candidates(children: List[Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Deduplicate children by normalized Action line.

    If the same normalized action appears multiple times, keep the child with the
    highest raw q target.  This matches the idea that the target estimates the best observed continuation after taking that action.
    """
    by_action: Dict[str, Dict[str, Any]] = {}
    skipped_invalid = 0

    for child in children:
        cand = build_candidate_from_child(child)
        if cand is None:
            skipped_invalid += 1
            continue

        key = cand["normalized_action"] or cand["action_text"]
        old = by_action.get(key)
        if old is None or cand["raw_target"] > old["raw_target"]:
            by_action[key] = cand

    return list(by_action.values()), skipped_invalid


def infer_shared_state_from_children(children: List[Any]) -> Tuple[List[Dict[str, str]], bool, int]:
    """Infer the decision state for parent.children from child.state.

    For QLASS TreeNode, a node stores the state *before its own action*. Thus, for a sibling candidate set parent.children, the common decision state is
    children[i].state, not parent.state.
    """
    states = [normalize_conversations(getattr(child, "state", None)) for child in children]
    signatures = [state_signature(s) for s in states]
    unique_signatures = set(signatures)
    mismatch = len(unique_signatures) > 1
    return states[0], mismatch, len(unique_signatures)


def iter_parent_nodes(root: Any) -> Iterable[Tuple[Any, int]]:
    """Depth-first traversal over parent nodes with their depth."""
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        yield node, depth
        children = getattr(node, "children", []) or []
        for child in reversed(children):
            stack.append((child, depth + 1))


def collect_raw_groups_from_tree(
    root: Any,
    task_id: Any,
    args: argparse.Namespace,
    tokenizer: Any = None,
    chat: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collect grouped, still-raw target examples from one MCTS tree."""
    # Match the current vanilla target construction.
    update_vanilla_rewards(root)
    root.update_q_values(gamma=args.gamma)

    groups: List[Dict[str, Any]] = []
    local_stats = Counter()
    depth_counter = Counter()
    trimming_stats = Counter()

    for parent, depth in iter_parent_nodes(root):
        children = getattr(parent, "children", []) or []
        if not children:
            continue

        local_stats["parent_nodes_with_children_raw"] += 1
        local_stats["total_candidates_raw"] += len(children)
        depth_counter[depth] += 1

        candidates, skipped_invalid = deduplicate_candidates(children)
        local_stats["skipped_invalid_children"] += skipped_invalid
        local_stats["total_candidates_after_dedup"] += len(candidates)

        if not candidates:
            local_stats["groups_k_eq_0_after_dedup"] += 1
            continue

        state_conversations, state_mismatch, n_unique_child_states = infer_shared_state_from_children(children)
        if state_mismatch:
            local_stats["groups_with_child_state_mismatch"] += 1

        candidate_action_msgs = [cand["action"] for cand in candidates]
        trim_info = None
        if tokenizer is not None and chat is not None:
            state_conversations, trim_info = trim_state_group_consistently(
                state_conversations=state_conversations,
                candidate_actions=candidate_action_msgs,
                tokenizer=tokenizer,
                chat=chat,
                max_prompt_tokens=args.max_prompt_tokens,
                keep_first_n=args.keep_first_n,
                min_tail_msgs=args.min_tail_msgs,
            )
            if trim_info["trimmed"]:
                trimming_stats["groups_trimmed"] += 1
            if trim_info["overflow_after_trim"]:
                trimming_stats["groups_overflow_after_trim"] += 1

        parent_action = action_to_message(getattr(parent, "action", ""))
        group = {
            "id": task_id,
            "parent_node_id": int(getattr(parent, "id", -1)),
            "parent_depth": int(depth),
            "parent_action": parent_action,
            "state_conversations": state_conversations,
            "candidates": candidates,
            "num_candidates_raw": int(len(children)),
            "num_candidates": int(len(candidates)),
            "state_mismatch_across_children": bool(state_mismatch),
            "num_unique_child_states": int(n_unique_child_states),
        }
        if trim_info is not None:
            group["trim_info"] = trim_info

        groups.append(group)

    local_stats["num_groups_after_dedup"] = len(groups)
    local_stats["num_groups_k_eq_1"] = sum(1 for g in groups if g["num_candidates"] == 1)
    local_stats["num_groups_k_ge_2"] = sum(1 for g in groups if g["num_candidates"] >= 2)
    local_stats["num_groups_saved_by_min_candidates"] = sum(
        1 for g in groups if g["num_candidates"] >= args.min_candidates
    )

    return groups, {
        "local_counts": dict(local_stats),
        "depth_counter": dict(depth_counter),
        "trimming_counts": dict(trimming_stats),
    }


def apply_tree_level_target_normalization(
    groups: List[Dict[str, Any]],
    normalization: str,
) -> Dict[str, float]:
    """Fill candidate['target'] from candidate['raw_target']."""
    raw_values = [cand["raw_target"] for group in groups for cand in group["candidates"]]
    if normalization == "raw":
        norm_values = raw_values
        min_val = min(raw_values) if raw_values else 0.0
        max_val = max(raw_values) if raw_values else 0.0
    elif normalization == "minmax":
        norm_values, min_val, max_val = minmax_normalize(raw_values)
    else:
        raise ValueError(f"Unsupported target normalization: {normalization}")

    idx = 0
    for group in groups:
        for cand in group["candidates"]:
            cand["target"] = float(norm_values[idx])
            idx += 1

    return {"tree_target_min_raw": float(min_val), "tree_target_max_raw": float(max_val)}


# -----------------------------
# Global statistics
# -----------------------------


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def compute_dataset_stats(
    all_groups: List[Dict[str, Any]],
    saved_groups: List[Dict[str, Any]],
    aggregate_counts: Counter,
    depth_counter: Counter,
    trimming_counter: Counter,
    target_zero_eps: float,
) -> Dict[str, Any]:
    candidate_counts_all = [g["num_candidates"] for g in all_groups]
    candidate_counts_saved = [g["num_candidates"] for g in saved_groups]
    targets_saved = [cand["target"] for g in saved_groups for cand in g["candidates"]]
    raw_targets_saved = [cand["raw_target"] for g in saved_groups for cand in g["candidates"]]

    within_stds = []
    within_raw_stds = []
    zero_var_groups = 0
    for group in saved_groups:
        vals = [cand["target"] for cand in group["candidates"]]
        raw_vals = [cand["raw_target"] for cand in group["candidates"]]
        std_val = safe_std(vals)
        within_stds.append(std_val)
        within_raw_stds.append(safe_std(raw_vals))
        if std_val <= target_zero_eps:
            zero_var_groups += 1

    total_raw = int(aggregate_counts.get("total_candidates_raw", 0))
    total_dedup = int(aggregate_counts.get("total_candidates_after_dedup", 0))

    return {
        "num_task_trees": int(aggregate_counts.get("num_task_trees", 0)),
        "num_parent_groups_total": len(all_groups),
        "num_parent_groups_saved": len(saved_groups),
        "num_parent_groups_k_eq_1": int(sum(1 for g in all_groups if g["num_candidates"] == 1)),
        "num_parent_groups_k_ge_2": int(sum(1 for g in all_groups if g["num_candidates"] >= 2)),
        "num_groups_with_child_state_mismatch": int(aggregate_counts.get("groups_with_child_state_mismatch", 0)),
        "num_groups_k_eq_0_after_dedup": int(aggregate_counts.get("groups_k_eq_0_after_dedup", 0)),
        "skipped_invalid_children": int(aggregate_counts.get("skipped_invalid_children", 0)),
        "total_candidates_raw": total_raw,
        "total_candidates_after_dedup": total_dedup,
        "unique_candidate_rate_after_dedup": float(total_dedup / total_raw) if total_raw else 0.0,
        "avg_candidates_per_group_all": safe_mean([float(x) for x in candidate_counts_all]),
        "median_candidates_per_group_all": percentile([float(x) for x in candidate_counts_all], 0.5),
        "max_candidates_per_group_all": int(max(candidate_counts_all) if candidate_counts_all else 0),
        "avg_candidates_per_group_saved": safe_mean([float(x) for x in candidate_counts_saved]),
        "median_candidates_per_group_saved": percentile([float(x) for x in candidate_counts_saved], 0.5),
        "max_candidates_per_group_saved": int(max(candidate_counts_saved) if candidate_counts_saved else 0),
        "target_min": float(min(targets_saved) if targets_saved else 0.0),
        "target_max": float(max(targets_saved) if targets_saved else 0.0),
        "target_mean": safe_mean([float(x) for x in targets_saved]),
        "target_std": safe_std([float(x) for x in targets_saved]),
        "raw_target_min": float(min(raw_targets_saved) if raw_targets_saved else 0.0),
        "raw_target_max": float(max(raw_targets_saved) if raw_targets_saved else 0.0),
        "raw_target_mean": safe_mean([float(x) for x in raw_targets_saved]),
        "raw_target_std": safe_std([float(x) for x in raw_targets_saved]),
        "avg_target_std_within_group": safe_mean(within_stds),
        "median_target_std_within_group": percentile(within_stds, 0.5),
        "avg_raw_target_std_within_group": safe_mean(within_raw_stds),
        "share_groups_with_zero_target_variance": float(zero_var_groups / len(saved_groups)) if saved_groups else 0.0,
        "num_groups_with_zero_target_variance": int(zero_var_groups),
        "depth_distribution_all_groups": {str(k): int(v) for k, v in sorted(depth_counter.items())},
        "trimming": {str(k): int(v) for k, v in sorted(trimming_counter.items())},
    }


def print_stats(stats: Dict[str, Any]) -> None:
    important_keys = [
        "num_task_trees",
        "num_parent_groups_total",
        "num_parent_groups_saved",
        "num_parent_groups_k_eq_1",
        "num_parent_groups_k_ge_2",
        "total_candidates_raw",
        "total_candidates_after_dedup",
        "unique_candidate_rate_after_dedup",
        "avg_candidates_per_group_all",
        "avg_candidates_per_group_saved",
        "target_min",
        "target_max",
        "target_mean",
        "target_std",
        "avg_target_std_within_group",
        "median_target_std_within_group",
        "share_groups_with_zero_target_variance",
        "num_groups_with_child_state_mismatch",
    ]
    print("\n[stats] Dueling dataset summary")
    for key in important_keys:
        print(f"  {key}: {stats.get(key)}")
    print(f"  depth_distribution_all_groups: {stats.get('depth_distribution_all_groups')}")
    print(f"  trimming: {stats.get('trimming')}")


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct grouped dueling Q-data from QLASS MCTS exploration trees."
    )
    parser.add_argument("--task", type=str, default="alfworld", help="Task name used for default output path.")
    parser.add_argument("--data_path", type=str, default=None, help="Directory with exploration .pkl files.")
    parser.add_argument("--combined_tree_path", type=str, default=None, help="Path to an existing combined_tree.pkl.")
    parser.add_argument("--rebuild_combined_tree", action="store_true", help="Rebuild combined_tree.pkl from --data_path .pkl files.")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save dueling_vanilla*.json.")
    parser.add_argument(
        "--target_normalization",
        type=str,
        choices=["raw", "minmax"],
        default="raw",
        help="raw keeps q targets unchanged; minmax applies per-tree min-max normalization like vanilla QLASS.",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount used by TreeNode.update_q_values.")
    parser.add_argument("--min_candidates", type=int, default=2, help="Save only groups with at least this many unique candidates.")
    parser.add_argument("--target_zero_eps", type=float, default=1e-12, help="Epsilon for zero within-group target variance stats.")

    # Optional group-consistent trimming.  If --model_name_or_path is omitted,
    # the script only stores raw conversations and trimming should be done later by the dueling Dataset/Collator.
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Tokenizer/chat-template path for optional trimming.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_tokens", type=int, default=3800)
    parser.add_argument("--keep_first_n", type=int, default=3)
    parser.add_argument("--min_tail_msgs", type=int, default=4)

    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON. Larger but easier to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_candidates < 1:
        raise ValueError("--min_candidates must be >= 1")

    if args.output_path is None:
        suffix = "raw" if args.target_normalization == "raw" else "minmax"
        args.output_path = f"data/train/{args.task}/explore/dueling_vanilla_{suffix}.json"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, chat = maybe_init_tokenizer(args)
    if tokenizer is None:
        print("[trim] --model_name_or_path is not set: storing untrimmed states.")
    else:
        print(
            f"[trim] Enabled group-consistent trimming with max_prompt_tokens={args.max_prompt_tokens}, "
            f"keep_first_n={args.keep_first_n}, min_tail_msgs={args.min_tail_msgs}"
        )

    tree_records = load_tree_records(args)
    print(f"[load] Found {len(tree_records)} task-tree records")

    all_groups: List[Dict[str, Any]] = []
    saved_groups: List[Dict[str, Any]] = []
    aggregate_counts = Counter()
    aggregate_depth_counter = Counter()
    aggregate_trimming_counter = Counter()

    for rec in tqdm(tree_records, desc="collect dueling groups"):
        task_id = rec.get("id")
        root = rec.get("tree")
        if root is None:
            continue

        aggregate_counts["num_task_trees"] += 1
        groups, local = collect_raw_groups_from_tree(
            root=root,
            task_id=task_id,
            args=args,
            tokenizer=tokenizer,
            chat=chat,
        )
        apply_tree_level_target_normalization(groups, args.target_normalization)

        for key, value in local["local_counts"].items():
            aggregate_counts[key] += value
        aggregate_depth_counter.update({int(k): int(v) for k, v in local["depth_counter"].items()})
        aggregate_trimming_counter.update(local["trimming_counts"])

        all_groups.extend(groups)
        saved_groups.extend(g for g in groups if g["num_candidates"] >= args.min_candidates)

    metadata = {
        "format": "qlass_dueling_vanilla_v1",
        "task": args.task,
        "target_normalization": args.target_normalization,
        "normalization_scope": "per_tree" if args.target_normalization == "minmax" else "none",
        "gamma": args.gamma,
        "min_candidates": args.min_candidates,
        "state_source": "children[0].state for each parent.children candidate set",
        "deduplication": "keep max raw_target per normalized Action line",
        "trimming_enabled": tokenizer is not None,
        "model_name_or_path_for_trimming": args.model_name_or_path,
    }

    stats = compute_dataset_stats(
        all_groups=all_groups,
        saved_groups=saved_groups,
        aggregate_counts=aggregate_counts,
        depth_counter=aggregate_depth_counter,
        trimming_counter=aggregate_trimming_counter,
        target_zero_eps=args.target_zero_eps,
    )
    metadata["stats"] = stats

    print_stats(stats)
    print(f"\n[save] Writing {len(saved_groups)} groups to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_groups, f, ensure_ascii=False, indent=2 if args.pretty else None)

    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    print(f"[save] Writing metadata/stats to {stats_path}")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
