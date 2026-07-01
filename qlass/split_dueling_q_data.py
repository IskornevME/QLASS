import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dueling_vanilla_*.json into train/val files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to dueling_vanilla_*.json.")
    parser.add_argument("--train_output_path", type=str, default=None, help="Optional train output path.")
    parser.add_argument("--val_output_path", type=str, default=None, help="Optional val output path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_by",
        type=str,
        default="id",
        choices=["id", "group"],
        help="Split by task/tree id to reduce leakage, or by individual groups.",
    )
    parser.add_argument(
        "--val_size_groups",
        type=int,
        default=1024,
        help="Target number of validation groups. With split_by=id this is approximate.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="Alternative to val_size_groups. If set, uses round(N * val_ratio).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    return parser.parse_args()


def target_std(ex: Dict[str, Any]) -> float:
    vals = [float(c["target"]) for c in ex.get("candidates", []) if "target" in c]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def summarize(name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    ks = [int(ex.get("num_candidates", len(ex.get("candidates", [])))) for ex in data]
    stds = [target_std(ex) for ex in data]
    targets = [float(c["target"]) for ex in data for c in ex.get("candidates", [])]
    by_depth = Counter(str(ex.get("parent_depth")) for ex in data)
    return {
        "name": name,
        "num_groups": len(data),
        "num_unique_ids": len({ex.get("id") for ex in data}),
        "num_candidates_total": sum(ks),
        "avg_k": (sum(ks) / len(ks)) if ks else 0.0,
        "max_k": max(ks) if ks else 0,
        "target_mean": (sum(targets) / len(targets)) if targets else 0.0,
        "share_zero_target_variance": (sum(s <= 1e-12 for s in stds) / len(stds)) if stds else 0.0,
        "avg_target_std_within_group": (sum(stds) / len(stds)) if stds else 0.0,
        "depth_distribution": dict(sorted(by_depth.items(), key=lambda kv: int(kv[0]) if kv[0] != "None" else -1)),
    }


def split_by_group(data: List[Dict[str, Any]], n_val: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    val_idx = set(indices[:n_val])
    train = [ex for i, ex in enumerate(data) if i not in val_idx]
    val = [ex for i, ex in enumerate(data) if i in val_idx]
    return train, val


def split_by_id(data: List[Dict[str, Any]], n_val: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    groups_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in data:
        groups_by_id[str(ex.get("id"))].append(ex)

    ids = list(groups_by_id.keys())
    rng.shuffle(ids)

    val_ids = set()
    val_count = 0
    for task_id in ids:
        val_ids.add(task_id)
        val_count += len(groups_by_id[task_id])
        if val_count >= n_val:
            break

    train = [ex for ex in data if str(ex.get("id")) not in val_ids]
    val = [ex for ex in data if str(ex.get("id")) in val_ids]
    return train, val


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {input_path}, got {type(data)}")
    if not data:
        raise ValueError(f"Input dataset is empty: {input_path}")

    if args.val_ratio is not None:
        if not (0.0 < args.val_ratio < 1.0):
            raise ValueError("--val_ratio must be in (0, 1)")
        n_val = max(1, round(len(data) * args.val_ratio))
    else:
        n_val = int(args.val_size_groups)

    if n_val <= 0 or n_val >= len(data):
        raise ValueError(f"Invalid validation size: {n_val}; dataset size={len(data)}")

    if args.split_by == "id":
        train, val = split_by_id(data, n_val=n_val, seed=args.seed)
    else:
        train, val = split_by_group(data, n_val=n_val, seed=args.seed)

    train_output = Path(args.train_output_path) if args.train_output_path else input_path.with_name(input_path.stem + "_train.json")
    val_output = Path(args.val_output_path) if args.val_output_path else input_path.with_name(input_path.stem + "_val.json")
    stats_output = input_path.with_name(input_path.stem + "_split_stats.json")

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    json_kwargs = {"ensure_ascii": False}
    if args.pretty:
        json_kwargs["indent"] = 2

    with train_output.open("w", encoding="utf-8") as f:
        json.dump(train, f, **json_kwargs)
    with val_output.open("w", encoding="utf-8") as f:
        json.dump(val, f, **json_kwargs)

    stats = {
        "input_path": str(input_path),
        "train_output_path": str(train_output),
        "val_output_path": str(val_output),
        "seed": args.seed,
        "split_by": args.split_by,
        "target_val_size_groups": n_val,
        "input": summarize("input", data),
        "train": summarize("train", train),
        "val": summarize("val", val),
    }
    with stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[split_dueling_q_data] Done")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
