
import json
import os
import re
import numpy as np

data_dir = './data/train/alfworld/'
# model_name = 'qlass-Llama-2-7b-chat-hf-alfworld-sft'
model_name = 'Qwen3-4B-Instruct-2507'
# inference_type = 'q_correction_memory_without_perturb'
# inference_type = 'q_without_perturb'
# inference_type = 'qwen_llama_qnet'
inference_type = 'qwen_qnet'
# eval_name = 'debug_bon2_test_run_aug_per_task_same_hparams_lambda0.0_gamma0.97_k10_thr0.75_per_task_no_terminal_override_aug2_percanon1_retrieved_thought_exact_only'
# eval_name = 'debug_bon2_test_run_aug_only_0_lambda0.0_gamma0.97_k10_thr0.75_global_no_terminal_override_aug2_percanon1_retrieved_thought_exact_only'
# eval_name = 'debug_bon2_test_run_0'
# eval_name = 'qwen_actor_llama_qnet_qnet_bon2_traj3_steps40_test_run0_no_memory_no_terminal_override_no_aug'
# eval_name = 'qwen_actor_qwen_llm_critic_llm_judge_bon2_traj3_steps40_test_run0_no_memory_no_terminal_override_no_aug'
# eval_name = 'qwen_actor_llama_qnet_aug_only_qnet_bon2_traj5_steps40_test_run0_memory_rewardterminal_only_lambda0.0_gamma0.97_k10_thr0.75_per_task_no_terminal_override_aug2_percanon1_retrieved_thought_exact_only'
eval_name = 'qwen_qnet_bon2_traj3_steps50_react50_test_runqwen_qnet_run1_no_memory_no_terminal_override_no_aug'
slice_num = 1
file_list = []
for i in range(slice_num):
    file_path = os.path.join(data_dir, model_name, inference_type, eval_name, f"{i}of{slice_num}_slices_bon_traj.jsonl")
    file_list.append(file_path)

from collections import Counter, defaultdict
from statistics import mean


def trajectory_score(traj):
    """Binary ALFWorld score, robust to reward=None."""
    success = traj.get("success")

    if success is not None:
        if isinstance(success, str):
            return float(success.strip().lower() == "true")
        return float(bool(success))

    reward = traj.get("reward")
    return 0.0 if reward is None else float(reward)


# Сначала объединяем все slices.
all_trajs = []

for path in file_list:
    with open(path, "r", encoding="utf-8") as f:
        slice_data = json.load(f)

    print(f"{path}: {len(slice_data)} trajectories")
    all_trajs.extend(slice_data)


# Группируем попытки по task id, поэтому n_trajs не нужно хардкодить.
# scores_by_task = defaultdict(list)

# for traj in all_trajs:
#     task_id = traj.get("id")
#     if task_id is None:
#         raise ValueError("Trajectory without task id.")

#     scores_by_task[str(task_id)].append(trajectory_score(traj))
trajs_by_task = defaultdict(list)

for traj in all_trajs:
    task_id = traj.get("id")

    if task_id is None:
        raise ValueError("Trajectory without task id.")

    trajs_by_task[str(task_id)].append(traj)


for task_id, trajs in trajs_by_task.items():
    trajs.sort(
        key=lambda traj: int(traj["attempt_id"])
    )

    attempt_ids = [
        int(traj["attempt_id"])
        for traj in trajs
    ]

    expected_ids = list(range(len(trajs)))

    if attempt_ids != expected_ids:
        raise ValueError(
            f"Unexpected attempt ids for task {task_id}: "
            f"{attempt_ids}"
        )


scores_by_task = {
    task_id: [
        trajectory_score(traj)
        for traj in trajs
    ]
    for task_id, trajs in trajs_by_task.items()
}


attempt_count_distribution = Counter(
    len(scores) for scores in scores_by_task.values()
)
print("attempt counts per task:", dict(attempt_count_distribution))

if len(attempt_count_distribution) != 1:
    raise ValueError(
        "Different tasks have different numbers of trajectories: "
        f"{dict(attempt_count_distribution)}"
    )

n_trajs = next(iter(attempt_count_distribution))
task_scores = list(scores_by_task.values())

first_score = mean(scores[0] for scores in task_scores)
last_score = mean(scores[-1] for scores in task_scores)
mean_score = mean(mean(scores) for scores in task_scores)
best_score = mean(max(scores) for scores in task_scores)

print(f"num tasks: {len(task_scores)}")
print(f"n_trajs: {n_trajs}")
print(f"first_trajectory_avg_reward: {first_score:.6f}")
print(f"last_trajectory_avg_reward:  {last_score:.6f}")
print(f"best_of_{n_trajs}_avg_reward: {best_score:.6f}")
print(f"mean_of_{n_trajs}_avg_reward: {mean_score:.6f}")

print("\nsuccess_rate_by_attempt:")
for attempt_id in range(n_trajs):
    attempt_scores = [scores[attempt_id] for scores in task_scores]
    print(
        f"  attempt {attempt_id}: "
        f"{mean(attempt_scores):.6f} "
        f"({int(sum(attempt_scores))}/{len(attempt_scores)})"
    )

print("\nrecovery_after_all_previous_failures:")

for attempt_id in range(1, n_trajs):
    eligible = [
        scores
        for scores in task_scores
        if max(scores[:attempt_id]) == 0
    ]

    recovered = sum(scores[attempt_id] > 0 for scores in eligible)

    rate = (recovered / len(eligible) if eligible else 0.0)

    print(
        f"  attempt {attempt_id}: "
        f"{rate:.6f} "
        f"({recovered}/{len(eligible)})"
    )

print("\nrecovery_after_previous_attempt_failure:")

for attempt_id in range(1, n_trajs):
    eligible = [
        scores
        for scores in task_scores
        if scores[attempt_id - 1] == 0
    ]

    recovered = sum(scores[attempt_id] > 0 for scores in eligible)

    rate = (recovered / len(eligible) if eligible else 0.0)

    print(
        f"  attempt {attempt_id}: "
        f"{rate:.6f} "
        f"({recovered}/{len(eligible)})"
    )

print("\nbest_of_k curve:")
for k in range(1, n_trajs + 1):
    best_of_k = mean(max(scores[:k]) for scores in task_scores)
    print(f"  best_of_{k}: {best_of_k:.6f}")


INVALID_ACTION_COMMAND = "__invalid_action__"


def normalize_action(action):
    return re.sub(r"\s+", " ", str(action).strip().lower())


def parse_logged_raw_action(raw_action):
    """
    Parse an action stored in old action_value_dict records.

    Mirrors AlfWorldEnv.parse_action():
      1. Qwen ReAct <action>...</action>
      2. legacy QLASS Action: ...
      3. malformed outputs -> one invalid-action sentinel
    """
    text = str(raw_action or "").strip()

    match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.DOTALL | re.IGNORECASE)

    if match:
        return normalize_action(match.group(1))

    match = re.search(r"Action:\s?(.*)", text, flags=re.DOTALL)

    if match:
        return normalize_action(match.group(1))

    return INVALID_ACTION_COMMAND


def iter_policy_actions_by_step(traj):
    """
    New runs:
        use parsed action_command from critic_attempt_trace.

    Old runs:
        reconstruct action_command from raw responses stored in action_value_dict.
    """
    trace = traj.get("critic_attempt_trace")

    if trace and trace.get("steps"):
        for step in trace["steps"]:
            actions = [normalize_action(candidate["action_command"])
                for candidate in step["candidates"]
                if candidate.get("candidate_source", "policy") == "policy"
            ]

            if actions:
                yield int(step["step_id"]), actions

        return

    # Backward-compatible path for old experiments.
    for step_id, step_candidates in enumerate(traj.get("action_value_dict", [])):
        actions = []

        for candidate in step_candidates:
            if candidate.get("candidate_source", "policy") != "policy":
                continue

            raw_action = candidate.get("action", "")

            actions.append(parse_logged_raw_action(raw_action))

        if actions:
            yield step_id, actions


diversity_by_step = defaultdict(list)

for trajs in trajs_by_task.values():
    for traj in trajs:
        for step_id, actions in (iter_policy_actions_by_step(traj)):
            num_candidates = len(actions)
            num_unique = len(set(actions))

            diversity_by_step[step_id].append(
                {
                    "num_candidates": num_candidates,
                    "num_unique": num_unique,
                }
            )

print("\npolicy_action_diversity_by_step:")

for step_id in sorted(diversity_by_step):
    rows = diversity_by_step[step_id]

    avg_candidates = mean(row["num_candidates"] for row in rows)

    avg_unique = mean(row["num_unique"] for row in rows)

    avg_unique_fraction = mean(row["num_unique"] / row["num_candidates"] for row in rows)

    all_same_rate = mean(row["num_unique"] == 1 for row in rows)

    print(
        f"  step {step_id}: "
        f"avg_candidates={avg_candidates:.3f}, "
        f"avg_unique={avg_unique:.3f}, "
        f"unique_fraction={avg_unique_fraction:.3f}, "
        f"all_same_rate={all_same_rate:.3f}, "
        f"n={len(rows)}"
    )


prompt_diagnostics_by_attempt = defaultdict(list)

for trajs in trajs_by_task.values():
    for traj in trajs:
        attempt_id = int(traj["attempt_id"])

        for step_candidates in traj.get("action_value_dict", []):
            for candidate in step_candidates:
                diagnostics = (candidate.get("critic_diagnostics") or {})

                if "qnet_prompt_tokens_with_attempt_memory" not in diagnostics:
                    continue

                prompt_diagnostics_by_attempt[attempt_id].append(diagnostics)

def print_percentiles(name, values):
    if not values:
        return

    values = np.asarray(values, dtype=float)

    print(
        f"    {name}: "
        f"p50={np.percentile(values, 50):.1f}, "
        f"p90={np.percentile(values, 90):.1f}, "
        f"p95={np.percentile(values, 95):.1f}, "
        f"p99={np.percentile(values, 99):.1f}, "
        f"max={values.max():.1f}"
    )

print("\ncritic_prompt_length_by_attempt:")

for attempt_id in sorted(prompt_diagnostics_by_attempt):
    rows = prompt_diagnostics_by_attempt[attempt_id]

    base_tokens = [row["qnet_prompt_tokens_without_attempt_memory"] for row in rows]
    full_tokens = [row["qnet_prompt_tokens_with_attempt_memory"] for row in rows]
    memory_tokens = [row["qnet_prompt_tokens_added_by_attempt_memory"] for row in rows]
    truncation_rate = mean(bool(row["qnet_prompt_would_truncate"]) for row in rows)
    retention = [row["attempt_memory_retention_ratio"] for row in rows if row["attempt_memory_used"]]

    print(f"  attempt {attempt_id}:")

    print_percentiles("without_memory_tokens", base_tokens)

    print_percentiles("with_memory_tokens_before_truncation", full_tokens)

    print_percentiles("memory_added_tokens", memory_tokens,)

    print(f"    truncation_rate={truncation_rate:.4f}")

    if retention:
        print_percentiles("memory_retention_ratio", retention)