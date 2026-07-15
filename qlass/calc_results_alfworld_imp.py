import argparse
import glob
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


SLICE_RE = re.compile(r"(?P<slice_id>\d+)of(?P<slice_num>\d+)_slices_(?P<sample_mode>.+?)_traj\.jsonl$")
_ACTION_RE = re.compile(r"Action:\s*(.*)", flags=re.IGNORECASE | re.DOTALL)


def normalize_env_action(text: object) -> str:
    if text is None:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    match = _ACTION_RE.search(s)
    if match:
        s = match.group(1).strip()

    # If model generated extra text after the action, keep only the first line.
    s = s.splitlines()[0].strip()

    # Normalize harmless formatting differences.
    s = s.rstrip(".")
    s = " ".join(s.lower().split())
    return s


def extract_env_action_from_candidate(candidate: dict) -> str:
    # Prefer parsed_action when present, but still parse it because old q_argmax
    # can store full Thought/Action text there.
    for key in ("parsed_action", "action", "action_text_for_env"):
        value = candidate.get(key)
        action = normalize_env_action(value)
        if action:
            return action
    return ""


def reward_to_float(value: Any) -> float:
    """Metric-safe reward conversion.

    In ALFWorld unfinished/non-successful trajectories may have reward=None.
    For success-rate / average-reward evaluation this should be treated as 0.0.
    """
    if value is None:
        return 0.0
    return float(value)


def bool_to_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def normalize_action(action: Any) -> str:
    if action is None:
        return ""
    return " ".join(str(action).strip().lower().split())


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list in {path}, got {type(data)}")
    return data


def discover_traj_files(output_dir: str, sample_mode: str, slice_num: Optional[int]) -> List[str]:
    if slice_num is not None:
        return [
            os.path.join(output_dir, f"{i}of{slice_num}_slices_{sample_mode}_traj.jsonl")
            for i in range(slice_num)
        ]

    pattern = os.path.join(output_dir, f"*of*_slices_{sample_mode}_traj.jsonl")
    return sorted(glob.glob(pattern))


def parse_slice_info(path: str) -> Tuple[Optional[int], Optional[int]]:
    match = SLICE_RE.search(os.path.basename(path))
    if not match:
        return None, None
    return int(match.group("slice_id")), int(match.group("slice_num"))


def iter_task_groups_by_file_order(
    files_with_data: List[Tuple[str, List[Dict[str, Any]]]],
    n_trajs: int,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Group trajectories into tasks by consecutive chunks within each slice file.

    This is the correct grouping for multi-worker q_guided_inference outputs, because
    the saved trajectory field `id` can be local/repeated across workers. Each worker
    writes trajectories in order: task_0 traj_0..n, task_1 traj_0..n, etc.
    """
    task_groups: List[Dict[str, Any]] = []

    for path, trajectories in files_with_data:
        slice_id, parsed_slice_num = parse_slice_info(path)

        remainder = len(trajectories) % n_trajs
        if remainder != 0:
            msg = (
                f"File {path} has {len(trajectories)} trajectories, which is not "
                f"divisible by n_trajs={n_trajs}. The last {remainder} trajectories "
                "will be ignored for task-level metrics."
            )
            if strict:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}")

        usable_len = (len(trajectories) // n_trajs) * n_trajs
        for local_task_index, start in enumerate(range(0, usable_len, n_trajs)):
            chunk = trajectories[start:start + n_trajs]
            # Keep a globally unique task key even if traj['id'] repeats across slices.
            task_key = f"slice{slice_id}_local{local_task_index}" if slice_id is not None else f"{os.path.basename(path)}_local{local_task_index}"
            task_groups.append(
                {
                    "task_key": task_key,
                    "source_file": path,
                    "slice_id": slice_id,
                    "slice_num": parsed_slice_num,
                    "local_task_index": local_task_index,
                    "stored_ids": [traj.get("id") for traj in chunk],
                    "trajectories": chunk,
                }
            )

    return task_groups


def get_step_candidates(traj: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    action_value_dict = traj.get("action_value_dict") or []
    if not isinstance(action_value_dict, list):
        return []
    return [step for step in action_value_dict if isinstance(step, list)]


def aggregate_saved_correction_stats(trajectories: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total_steps = 0
    changed_steps = 0
    num_trajs_with_stats = 0

    for traj in trajectories:
        stats = traj.get("q_adv_correction_stats")
        if not isinstance(stats, dict):
            continue

        q_adv_steps = int(stats.get("num_q_adv_steps", 0) or 0)
        changed = int(stats.get("num_raw_vs_selected_changed", 0) or 0)

        if q_adv_steps > 0:
            num_trajs_with_stats += 1
            total_steps += q_adv_steps
            changed_steps += changed

    return {
        "source": "q_adv_correction_stats",
        "num_trajs_with_stats": num_trajs_with_stats,
        "num_q_adv_steps": total_steps,
        "num_raw_vs_selected_changed": changed_steps,
        "raw_vs_selected_changed_rate": safe_div(changed_steps, total_steps),
    }


def recover_changed_steps_from_action_value_dict(
    trajectories: Iterable[Dict[str, Any]],
    max_examples: int = 20,
) -> Dict[str, Any]:
    """Recover raw-vs-selected correction stats from saved per-step candidate records.

    This works if q_guided_inference.py saved flags inside each candidate record:
      - selected_by_raw_q
      - selected_by_final_policy or selected
    If raw flag is missing, fallback to max(selection_score_raw), but such steps are
    counted separately as `raw_recovered_by_score_steps`.
    """
    total_steps = 0
    changed_steps = 0
    missing_selected_steps = 0
    raw_recovered_by_score_steps = 0
    missing_raw_steps = 0

    examples: List[Dict[str, Any]] = []

    for traj_index, traj in enumerate(trajectories):
        dataset = traj.get("dataset")
        task_id = traj.get("id")

        for step_idx, step_candidates in enumerate(get_step_candidates(traj)):
            raw_candidate = None
            selected_candidate = None

            for cand in step_candidates:
                if cand.get("selected_by_raw_q"):
                    raw_candidate = cand
                if cand.get("selected_by_final_policy") or cand.get("selected"):
                    selected_candidate = cand

            if raw_candidate is None:
                scored = [
                    cand for cand in step_candidates
                    if cand.get("selection_score_raw") is not None
                ]
                if scored:
                    raw_candidate = max(scored, key=lambda c: float(c.get("selection_score_raw")))
                    raw_recovered_by_score_steps += 1
                else:
                    missing_raw_steps += 1
                    continue

            if selected_candidate is None:
                missing_selected_steps += 1
                continue

            raw_action = normalize_action(
                raw_candidate.get("parsed_action", raw_candidate.get("action"))
            )
            selected_action = normalize_action(
                selected_candidate.get("parsed_action", selected_candidate.get("action"))
            )

            total_steps += 1
            changed = raw_action != selected_action
            if changed:
                changed_steps += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "traj_index": traj_index,
                            "dataset": dataset,
                            "task_id": task_id,
                            "step_idx": step_idx,
                            "raw_action": raw_action,
                            "selected_action": selected_action,
                            "raw_idx": raw_candidate.get("idx"),
                            "selected_idx": selected_candidate.get("idx"),
                            "raw_score": raw_candidate.get("selection_score_raw"),
                            "raw_corrected_score": raw_candidate.get("corrected_score"),
                            "selected_raw_score": selected_candidate.get("selection_score_raw"),
                            "selected_corrected_score": selected_candidate.get("corrected_score"),
                        }
                    )

    return {
        "source": "action_value_dict",
        "num_q_adv_steps": total_steps,
        "num_raw_vs_selected_changed": changed_steps,
        "raw_vs_selected_changed_rate": safe_div(changed_steps, total_steps),
        "missing_raw_steps": missing_raw_steps,
        "missing_selected_steps": missing_selected_steps,
        "raw_recovered_by_score_steps": raw_recovered_by_score_steps,
        "examples": examples,
    }


def aggregate_unique_candidates_per_step(trajectories, args):
    num_steps = 0

    total_candidate_records = 0
    total_unique_env_actions = 0

    min_unique_env_actions = None
    max_unique_env_actions = None

    steps_with_all_duplicate_actions = 0
    steps_with_full_bon_unique_actions = 0

    for traj in trajectories:
        for step_candidates in traj.get("action_value_dict", []):
            if not step_candidates:
                continue

            num_steps += 1
            total_candidate_records += len(step_candidates)

            unique_env_actions = {
                extract_env_action_from_candidate(candidate)
                for candidate in step_candidates
            }
            unique_env_actions.discard("")

            n_unique = len(unique_env_actions)
            total_unique_env_actions += n_unique

            if min_unique_env_actions is None:
                min_unique_env_actions = n_unique
                max_unique_env_actions = n_unique
            else:
                min_unique_env_actions = min(min_unique_env_actions, n_unique)
                max_unique_env_actions = max(max_unique_env_actions, n_unique)

            if n_unique == 1:
                steps_with_all_duplicate_actions += 1
            if n_unique >= args.expected_bon:
                steps_with_full_bon_unique_actions += 1

    if num_steps == 0:
        return {
            "source": "action_value_dict",
            "num_steps_with_candidates": 0,
            "avg_candidate_records_per_step": 0.0,
            "avg_unique_candidates_per_step": 0.0,
            "avg_unique_env_actions_per_step": 0.0,
            "min_unique_env_actions_per_step": None,
            "max_unique_env_actions_per_step": None,
            "share_steps_with_all_duplicate_actions": 0.0,
            "share_steps_with_full_bon_unique_actions": 0.0,
        }

    avg_unique_env_actions = total_unique_env_actions / num_steps

    return {
        "source": "action_value_dict",
        "num_steps_with_candidates": num_steps,

        # Old value, useful for sanity check / compute budget.
        "avg_candidate_records_per_step": total_candidate_records / num_steps,

        # Keep old key name, but now make it mean real unique env actions.
        "avg_unique_candidates_per_step": avg_unique_env_actions,
        "avg_unique_env_actions_per_step": avg_unique_env_actions,

        "min_unique_env_actions_per_step": min_unique_env_actions,
        "max_unique_env_actions_per_step": max_unique_env_actions,
        "share_steps_with_all_duplicate_actions": steps_with_all_duplicate_actions / num_steps,
        "share_steps_with_full_bon_unique_actions": steps_with_full_bon_unique_actions / num_steps,
    }


def print_metric_block(title: str, metrics: Dict[str, Any]) -> None:
    print()
    print(f"[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate ALFWorld QLASS/Q-Adv multi-slice inference results."
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--slice_num", type=int, default=None)
    parser.add_argument("--sample_mode", type=str, default="bon")
    parser.add_argument("--expected_bon", type=int, default=2)
    parser.add_argument("--n_trajs", type=int, default=3)

    # Backward-compatible old-style path construction.
    parser.add_argument("--data_dir", type=str, default="./data/train/alfworld/")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--inference_type", type=str, default=None)
    parser.add_argument("--eval_name", type=str, default=None)

    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save_summary_json", type=str, default=None)
    parser.add_argument("--max_changed_examples", type=int, default=20)
    args = parser.parse_args()

    if args.n_trajs <= 0:
        raise ValueError(f"--n_trajs must be positive, got {args.n_trajs}")

    if args.output_dir is None:
        missing = [
            name for name, value in [
                ("--model_name", args.model_name),
                ("--inference_type", args.inference_type),
                ("--eval_name", args.eval_name),
            ] if value is None
        ]
        if missing:
            raise ValueError(
                "Either pass --output_dir or old-style args. Missing: " + ", ".join(missing)
            )
        args.output_dir = os.path.join(
            args.data_dir, args.model_name, args.inference_type, args.eval_name
        )

    candidate_files = discover_traj_files(args.output_dir, args.sample_mode, args.slice_num)
    existing_files = [path for path in candidate_files if os.path.exists(path)]
    missing_files = [path for path in candidate_files if not os.path.exists(path)]

    if missing_files:
        print("[WARN] Missing trajectory files:")
        for path in missing_files:
            print(f"  {path}")
        if args.strict:
            raise FileNotFoundError("Some expected trajectory files are missing.")

    if not existing_files:
        raise FileNotFoundError(
            f"No trajectory files found in output_dir={args.output_dir} "
            f"for sample_mode={args.sample_mode}, slice_num={args.slice_num}"
        )

    files_with_data: List[Tuple[str, List[Dict[str, Any]]]] = []
    all_trajs: List[Dict[str, Any]] = []

    print("[FILES]")
    for path in existing_files:
        data = load_json(path)
        slice_id, parsed_slice_num = parse_slice_info(path)
        num_complete_tasks = len(data) // args.n_trajs
        remainder = len(data) % args.n_trajs
        files_with_data.append((path, data))
        all_trajs.extend(data)
        print(
            f"  {path}: {len(data)} trajectories, "
            f"complete_tasks_by_order={num_complete_tasks}, remainder={remainder}, "
            f"slice_id={slice_id}, slice_num={parsed_slice_num}"
        )

    rewards = [reward_to_float(traj.get("reward")) for traj in all_trajs]
    successes = [bool_to_float(traj.get("success")) for traj in all_trajs]

    trajectory_metrics = {
        "num_files": len(existing_files),
        "num_trajectories": len(all_trajs),
        "expected_tasks_if_complete": safe_div(len(all_trajs), args.n_trajs),
        "avg_reward_over_all_trajectories": safe_div(sum(rewards), len(rewards)),
        "success_rate_over_all_trajectories_by_reward": safe_div(
            sum(1.0 for r in rewards if r == 1.0), len(rewards)
        ),
        "success_rate_over_all_trajectories_by_success_flag": safe_div(sum(successes), len(successes)),
    }
    print_metric_block("TRAJECTORY_LEVEL", trajectory_metrics)

    task_groups = iter_task_groups_by_file_order(files_with_data, args.n_trajs, strict=args.strict)

    first_rewards: List[float] = []
    best_rewards: List[float] = []
    mean_rewards: List[float] = []

    for group in task_groups:
        cur_rewards = [reward_to_float(traj.get("reward")) for traj in group["trajectories"]]
        first_rewards.append(cur_rewards[0])
        best_rewards.append(max(cur_rewards))
        mean_rewards.append(sum(cur_rewards) / len(cur_rewards))

    task_metrics = {
        "grouping": "by_consecutive_chunks_inside_each_slice_file",
        "num_task_groups_total": len(task_groups),
        "n_trajs": args.n_trajs,
        "first_trajectory_avg_reward": safe_div(sum(first_rewards), len(first_rewards)),
        f"best_of_{args.n_trajs}_avg_reward": safe_div(sum(best_rewards), len(best_rewards)),
        f"mean_of_{args.n_trajs}_avg_reward": safe_div(sum(mean_rewards), len(mean_rewards)),
    }
    print_metric_block("TASK_LEVEL_BY_SLICE_ORDER", task_metrics)

    saved_corr = aggregate_saved_correction_stats(all_trajs)
    recovered_corr = recover_changed_steps_from_action_value_dict(
        all_trajs,
        max_examples=args.max_changed_examples,
    )
    unique_candidates_stats = aggregate_unique_candidates_per_step(all_trajs, args)

    print_metric_block("CORRECTION_STATS_SAVED_PER_TRAJECTORY", saved_corr)
    print_metric_block(
        "CORRECTION_STATS_RECOVERED_FROM_ACTION_VALUE_DICT",
        {k: v for k, v in recovered_corr.items() if k != "examples"},
    )
    print_metric_block("UNIQUE_CANDIDATES_PER_STEP", unique_candidates_stats)

    if recovered_corr["examples"]:
        print()
        print("[RECOVERED_CHANGED_ACTION_EXAMPLES]")
        for ex in recovered_corr["examples"]:
            print(
                f"traj_index={ex['traj_index']} task_id={ex['task_id']} step={ex['step_idx']} "
                f"raw={ex['raw_action']!r} -> selected={ex['selected_action']!r} "
                f"raw_score={ex['raw_score']} raw_corrected={ex['raw_corrected_score']} "
                f"selected_raw={ex['selected_raw_score']} selected_corrected={ex['selected_corrected_score']}"
            )

    summary = {
        "output_dir": args.output_dir,
        "files": existing_files,
        "trajectory_metrics": trajectory_metrics,
        "task_metrics": task_metrics,
        "saved_correction_stats": saved_corr,
        "recovered_correction_stats": recovered_corr,
        "unique_candidates_per_step": unique_candidates_stats,
    }

    if args.save_summary_json:
        with open(args.save_summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print()
        print(f"[SAVED] {args.save_summary_json}")


if __name__ == "__main__":
    main()
