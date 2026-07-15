
import json
import os

data_dir = './data/train/alfworld/'
model_name = 'qlass-Llama-2-7b-chat-hf-alfworld-sft'
# inference_type = 'q_correction_memory_without_perturb'
inference_type = 'q_without_perturb'
# eval_name = 'debug_bon2_test_run_aug_only_0_lambda0.0_gamma0.97_k10_thr0.75_global_no_terminal_override_aug2_percanon1_retrieved_thought_exact_only'
eval_name = 'debug_bon2_test_run_0'
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
scores_by_task = defaultdict(list)

for traj in all_trajs:
    task_id = traj.get("id")
    if task_id is None:
        raise ValueError("Trajectory without task id.")

    scores_by_task[str(task_id)].append(trajectory_score(traj))


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

print("\nbest_of_k curve:")
for k in range(1, n_trajs + 1):
    best_of_k = mean(max(scores[:k]) for scores in task_scores)
    print(f"  best_of_{k}: {best_of_k:.6f}")
