#!/usr/bin/env python3
"""
Calculate SciWorld evaluation metrics from QLASS trajectory files.

QLASS writes trajectory files with a .jsonl suffix, but the current inference
script stores one JSON array in each file. This script supports both that format
and ordinary line-delimited JSON for robustness.

Examples
--------
Single slice:
    python qlass/calc_results_sciworld.py \
        data/train/sciworld/.../0of1_slices_bon_traj.jsonl

Several slices:
    python qlass/calc_results_sciworld.py \
        data/train/sciworld/.../*_traj.jsonl

Directory:
    python qlass/calc_results_sciworld.py data/train/sciworld/.../run_dir
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float."""
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool:
    """Parse bool-like values used in saved trajectory records."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", "", "none", "null"}:
            return False

    return bool(value)


def load_records(path: Path) -> list[dict[str, Any]]:
    """
    Load either:
      1) the current QLASS format: one JSON array per file; or
      2) ordinary JSONL: one JSON object per line.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}: JSONL record on line {line_number} "
                    f"must be an object, got {type(record).__name__}."
                )
            records.append(record)

        return records

    if isinstance(payload, list):
        if not all(isinstance(record, dict) for record in payload):
            raise ValueError(f"{path}: every item in the JSON array must be an object.")
        return payload

    if isinstance(payload, dict):
        return [payload]

    raise ValueError(
        f"{path}: expected a JSON array, object, or JSONL records; "
        f"got {type(payload).__name__}."
    )


def resolve_input_paths(inputs: Iterable[str]) -> list[Path]:
    """Resolve explicit files, directories, and unexpanded glob patterns."""
    resolved: list[Path] = []

    for raw_input in inputs:
        path = Path(raw_input)

        if path.is_file():
            resolved.append(path)
            continue

        if path.is_dir():
            resolved.extend(sorted(path.glob("*_traj.jsonl")))
            continue

        matches = [Path(match) for match in sorted(glob.glob(raw_input))]
        resolved.extend(match for match in matches if match.is_file())

    # Preserve order while removing duplicates.
    unique_paths = list(OrderedDict((str(path.resolve()), path) for path in resolved).values())

    if not unique_paths:
        raise FileNotFoundError(
            "No trajectory files found. Pass one or more *_traj.jsonl files, "
            "a directory containing them, or a glob pattern."
        )

    return unique_paths


def trajectory_reward(traj: dict[str, Any]) -> float:
    """
    Return the SciWorld reward saved by q_guided_inference.py.

    In the current SciWorldEnv implementation this is raw_score-scale data
    (normally 0..1), not the 0..100 paper display scale.
    """
    return safe_float(traj.get("reward"), default=0.0)


def trajectory_success(traj: dict[str, Any]) -> bool:
    """Return full task completion status."""
    if "success" in traj:
        return parse_bool(traj.get("success"))

    # Backward-compatible fallback for old records without an explicit flag.
    return math.isclose(trajectory_reward(traj), 1.0, rel_tol=0.0, abs_tol=1e-12)


def format_reward(value: float, paper_scale: float) -> str:
    return f"{value:.6f} (paper scale: {value * paper_scale:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-attempt, mean-of-N, and best-of-N SciWorld metrics "
            "from QLASS trajectory files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Trajectory files, directories, or glob patterns. "
            "Current QLASS files usually match *_traj.jsonl."
        ),
    )
    parser.add_argument(
        "--expected-n-trajs",
        type=int,
        default=None,
        help="Optionally require exactly this many trajectories for every task.",
    )
    parser.add_argument(
        "--paper-scale",
        type=float,
        default=100.0,
        help=(
            "Multiplier used only for display alongside raw rewards. "
            "SciWorld paper tables use a 0..100 scale; default: 100."
        ),
    )
    args = parser.parse_args()

    if args.expected_n_trajs is not None and args.expected_n_trajs <= 0:
        parser.error("--expected-n-trajs must be positive.")

    if args.paper_scale <= 0:
        parser.error("--paper-scale must be positive.")

    paths = resolve_input_paths(args.inputs)

    all_trajs: list[dict[str, Any]] = []
    for path in paths:
        records = load_records(path)
        print(f"{path}: {len(records)} trajectories")
        all_trajs.extend(records)

    if not all_trajs:
        raise ValueError("The resolved files contain no trajectories.")

    # Ordered grouping preserves attempt order as written by q_guided_inference.py.
    trajectories_by_task: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    for index, traj in enumerate(all_trajs):
        task_id = traj.get("id")
        if task_id is None:
            raise ValueError(f"Trajectory index {index} does not contain task id field 'id'.")

        trajectories_by_task.setdefault(str(task_id), []).append(traj)

    attempt_count_distribution = Counter(
        len(task_trajectories)
        for task_trajectories in trajectories_by_task.values()
    )

    print("\nattempt counts per task:", dict(sorted(attempt_count_distribution.items())))

    if len(attempt_count_distribution) != 1:
        raise ValueError(
            "Different tasks have different numbers of trajectories. "
            "Per-attempt metrics would be ambiguous: "
            f"{dict(sorted(attempt_count_distribution.items()))}"
        )

    n_trajs = next(iter(attempt_count_distribution))

    if args.expected_n_trajs is not None and n_trajs != args.expected_n_trajs:
        raise ValueError(
            f"Expected {args.expected_n_trajs} trajectories per task, found {n_trajs}."
        )

    rewards_by_task = [
        [trajectory_reward(traj) for traj in task_trajectories]
        for task_trajectories in trajectories_by_task.values()
    ]
    successes_by_task = [
        [trajectory_success(traj) for traj in task_trajectories]
        for task_trajectories in trajectories_by_task.values()
    ]

    num_tasks = len(trajectories_by_task)
    num_trajectories = len(all_trajs)

    first_reward = mean(rewards[0] for rewards in rewards_by_task)
    last_reward = mean(rewards[-1] for rewards in rewards_by_task)
    mean_reward = mean(mean(rewards) for rewards in rewards_by_task)
    best_reward = mean(max(rewards) for rewards in rewards_by_task)

    first_success = mean(float(successes[0]) for successes in successes_by_task)
    last_success = mean(float(successes[-1]) for successes in successes_by_task)
    mean_success = mean(
        mean(float(success) for success in successes)
        for successes in successes_by_task
    )
    best_success = mean(
        float(any(successes))
        for successes in successes_by_task
    )

    print("\n=== SciWorld summary ===")
    print(f"num tasks:        {num_tasks}")
    print(f"n_trajs:          {n_trajs}")
    print(f"num trajectories: {num_trajectories}")

    print("\n=== Reward metrics ===")
    print(f"first_trajectory_avg_reward: {format_reward(first_reward, args.paper_scale)}")
    print(f"last_trajectory_avg_reward:  {format_reward(last_reward, args.paper_scale)}")
    print(f"mean_of_{n_trajs}_avg_reward:       {format_reward(mean_reward, args.paper_scale)}")
    print(f"best_of_{n_trajs}_avg_reward:       {format_reward(best_reward, args.paper_scale)}")

    print("\n=== Full-success metrics ===")
    print(
        f"first_trajectory_success_rate: {first_success:.6f} "
        f"({sum(int(successes[0]) for successes in successes_by_task)}/{num_tasks})"
    )
    print(
        f"last_trajectory_success_rate:  {last_success:.6f} "
        f"({sum(int(successes[-1]) for successes in successes_by_task)}/{num_tasks})"
    )
    print(f"mean_of_{n_trajs}_success_rate:       {mean_success:.6f}")
    print(
        f"best_of_{n_trajs}_success_rate:       {best_success:.6f} "
        f"({sum(int(any(successes)) for successes in successes_by_task)}/{num_tasks})"
    )

    print("\n=== Metrics by attempt ===")
    for attempt_id in range(n_trajs):
        attempt_rewards = [
            rewards[attempt_id]
            for rewards in rewards_by_task
        ]
        attempt_successes = [
            successes[attempt_id]
            for successes in successes_by_task
        ]

        avg_reward = mean(attempt_rewards)
        success_count = sum(int(success) for success in attempt_successes)
        success_rate = success_count / num_tasks

        print(
            f"attempt {attempt_id}: "
            f"avg_reward={format_reward(avg_reward, args.paper_scale)}, "
            f"full_success_rate={success_rate:.6f} "
            f"({success_count}/{num_tasks})"
        )

    print("\n=== Best-of-k curves ===")
    for k in range(1, n_trajs + 1):
        best_of_k_reward = mean(
            max(rewards[:k])
            for rewards in rewards_by_task
        )
        best_of_k_success_count = sum(
            int(any(successes[:k]))
            for successes in successes_by_task
        )
        best_of_k_success_rate = best_of_k_success_count / num_tasks

        print(
            f"best_of_{k}: "
            f"avg_reward={format_reward(best_of_k_reward, args.paper_scale)}, "
            f"full_success_rate={best_of_k_success_rate:.6f} "
            f"({best_of_k_success_count}/{num_tasks})"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
