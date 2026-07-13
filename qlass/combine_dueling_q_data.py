#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple dueling_vanilla_raw.json datasets with global dedup."
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="Input dueling_vanilla_raw.json files.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output combined dueling_vanilla_raw.json.",
    )
    parser.add_argument(
        "--stats_output_path",
        default=None,
        help="Optional output stats path. Default: <output>.stats.json",
    )
    parser.add_argument(
        "--min_candidates",
        type=int,
        default=2,
        help="Drop groups with fewer than this many candidates after action dedup.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON. Useful for debugging, slower and larger.",
    )
    return parser.parse_args()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def get_state(ex: Dict[str, Any]) -> Any:
    # Keep this tolerant because slightly different builders may use different field names.
    for key in ("state", "state_conversations", "parent_state", "messages", "conversations"):
        if key in ex:
            return ex[key]
    raise KeyError(f"Could not find state field in example keys={list(ex.keys())}")


def candidate_text(cand: Dict[str, Any]) -> str:
    for key in ("action", "text", "value", "response", "completion"):
        if key in cand:
            value = cand[key]
            if isinstance(value, dict):
                value = value.get("value", value.get("content", ""))
            return str(value)
    return ""


def normalize_action_text(text: Any) -> str:
    if text is None:
        return ""

    if isinstance(text, dict):
        text = text.get("value", text.get("content", ""))

    text = str(text).strip()
    if not text:
        return ""

    # Prefer the final Action: segment if the model generated Thought + Action.
    match = re.search(r"Action:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    text = lines[0].rstrip(".").lower()
    return " ".join(text.split())


def candidate_target(cand: Dict[str, Any]) -> float:
    if "target" in cand:
        return float(cand["target"])
    if "raw_target" in cand:
        return float(cand["raw_target"])
    raise KeyError(f"Candidate has no target/raw_target. keys={list(cand.keys())}")


def target_std(ex: Dict[str, Any]) -> float:
    vals = [candidate_target(c) for c in ex.get("candidates", [])]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def normalize_candidates(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Deduplicate candidates inside one group by normalized action.

    If the same normalized action appears multiple times, keep the candidate
    with the largest target. This matches the usual optimistic target logic.
    """
    by_action: Dict[str, Dict[str, Any]] = {}

    for cand in candidates:
        action_key = normalize_action_text(candidate_text(cand))
        if not action_key:
            continue

        if action_key not in by_action:
            by_action[action_key] = cand
            continue

        if candidate_target(cand) > candidate_target(by_action[action_key]):
            by_action[action_key] = cand

    return by_action


def make_group_key(ex: Dict[str, Any], by_action: Dict[str, Dict[str, Any]]) -> Tuple[str, Tuple[str, ...]]:
    state_hash = stable_hash(get_state(ex))
    action_set = tuple(sorted(by_action.keys()))
    return state_hash, action_set


def merge_duplicate_examples(
    old_ex: Dict[str, Any],
    new_ex: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge duplicate groups with the same state and same normalized action set.

    For every action, keep the candidate with the highest target.
    For group-level metadata, keep the example with larger within-group target std,
    because it usually carries stronger ranking signal.
    """
    merged_by_action: Dict[str, Dict[str, Any]] = {}

    for ex in (old_ex, new_ex):
        for cand in ex.get("candidates", []):
            action_key = normalize_action_text(candidate_text(cand))
            if not action_key:
                continue

            if action_key not in merged_by_action:
                merged_by_action[action_key] = cand
            elif candidate_target(cand) > candidate_target(merged_by_action[action_key]):
                merged_by_action[action_key] = cand

    base = old_ex if target_std(old_ex) >= target_std(new_ex) else new_ex
    merged = dict(base)
    merged["candidates"] = [merged_by_action[a] for a in sorted(merged_by_action.keys())]
    merged["num_candidates"] = len(merged["candidates"])
    return merged


def summarize(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    ks = [len(ex.get("candidates", [])) for ex in data]
    stds = [target_std(ex) for ex in data]
    targets = [candidate_target(c) for ex in data for c in ex.get("candidates", [])]
    depths = Counter(str(ex.get("parent_depth")) for ex in data)

    return {
        "num_groups": len(data),
        "num_unique_ids": len({str(ex.get("id")) for ex in data}),
        "num_candidates_total": sum(ks),
        "avg_candidates_per_group": (sum(ks) / len(ks)) if ks else 0.0,
        "median_candidates_per_group": sorted(ks)[len(ks) // 2] if ks else 0,
        "max_candidates_per_group": max(ks) if ks else 0,
        "target_min": min(targets) if targets else None,
        "target_max": max(targets) if targets else None,
        "target_mean": (sum(targets) / len(targets)) if targets else 0.0,
        "share_groups_with_zero_target_variance": (
            sum(s <= 1e-12 for s in stds) / len(stds)
        ) if stds else 0.0,
        "num_groups_with_zero_target_variance": sum(s <= 1e-12 for s in stds),
        "avg_target_std_within_group": (sum(stds) / len(stds)) if stds else 0.0,
        "depth_distribution": dict(
            sorted(depths.items(), key=lambda kv: int(kv[0]) if kv[0] != "None" else -1)
        ),
    }


def main() -> None:
    args = parse_args()

    dedup: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
    input_stats = []
    total_loaded = 0
    skipped_too_few_candidates = 0
    duplicate_groups_merged = 0

    for input_path_str in args.input_paths:
        input_path = Path(input_path_str)
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {input_path}, got {type(data)}")

        input_stats.append(
            {
                "path": str(input_path),
                "num_groups": len(data),
                "summary": summarize(data),
            }
        )
        total_loaded += len(data)

        for ex in data:
            candidates = ex.get("candidates", [])
            if not isinstance(candidates, list):
                raise ValueError(f"Expected candidates list, got {type(candidates)}")

            by_action = normalize_candidates(candidates)
            if len(by_action) < args.min_candidates:
                skipped_too_few_candidates += 1
                continue

            normalized_ex = dict(ex)
            normalized_ex["candidates"] = [by_action[a] for a in sorted(by_action.keys())]
            normalized_ex["num_candidates"] = len(normalized_ex["candidates"])

            key = make_group_key(normalized_ex, by_action)

            if key in dedup:
                dedup[key] = merge_duplicate_examples(dedup[key], normalized_ex)
                duplicate_groups_merged += 1
            else:
                dedup[key] = normalized_ex

    combined = list(dedup.values())

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_kwargs = {"ensure_ascii": False}
    if args.pretty:
        json_kwargs["indent"] = 2

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, **json_kwargs)

    stats_output_path = (
        Path(args.stats_output_path)
        if args.stats_output_path is not None
        else output_path.with_suffix(output_path.suffix + ".stats.json")
    )

    stats = {
        "input_paths": args.input_paths,
        "output_path": str(output_path),
        "min_candidates": args.min_candidates,
        "total_loaded_groups": total_loaded,
        "output_groups_after_global_dedup": len(combined),
        "duplicate_groups_merged": duplicate_groups_merged,
        "skipped_too_few_candidates_after_action_dedup": skipped_too_few_candidates,
        "inputs": input_stats,
        "combined": summarize(combined),
    }

    with stats_output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[combine_dueling_q_data] Done")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
