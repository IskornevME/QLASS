# Semantic perturbation functions on original trajectories
# from openai import OpenAI
import openai
import json
import os
import argparse
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import sys
import os
root_dir = os.getcwd()
from pathlib import Path
sys.path.append(root_dir)
sys.path.append((Path(root_dir).parent))
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pickle
import random
import copy
# from model.q_network import QNet
from qlass.q_network import QNet

try:
    from eval.hotpotqa.zeno_build.models import lm_config
except:
    from zeno_build.models import lm_config

import random
import torch
import numpy as np
from rich import print
from qlass.explore_sft_agent import TreeNode


import qlass.tasks as tasks
import qlass.agents as agents
import qlass.envs as envs
from typing import List, Dict, Any
import logging
from colorama import Fore
from eval_agent.prompt.templates import *
from eval_agent.prompt.instructions import *

logger = logging.getLogger("agent_frame")
openai.api_key = os.environ["OPENAI_API_KEY"]
model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-3.5-turbo")
eval_model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-4-1106-preview")
engine = 'gpt-3.5-turbo'
perturb_prompt = """
Please paraphrase the following text:
"""

def perturb_messages_for_webshop(instruction, messages):

    message = messages[0]['content'].split('here is the task.\nWebShop [SEP] Instruction: [SEP] ')[-1].split(' [SEP] Search')[0].strip()
    predictions = generate_from_openai_chat_completion(
            full_contexts=[chat_prompt.ChatMessages(
                [{"role": "user", "content": f"Paraphrase the task:\n\n{message}\n\n"}]
            )],
            model_config=model_config,
            temperature=0.7+0.1*random.random(),
            max_tokens=200,
            tqdm=False
        )
    # print("predictions:", predictions)
    task = f"WebShop [SEP] Instruction: [SEP] {predictions[0].strip()} [SEP] Search"
    messages[0]['content'] = PROMPT_WITHOUT_ICL_TEMPLATE.format(instruction=instruction, task=task)
    print("messages[0]:", messages[0])

    return messages

def perturb_messages_for_sciworld(instruction, messages):
    raise NotImplementedError("We do not support the task to perturb messages: sciworld")



def perturb_messages_for_alfworld(instruction, messages):
    raise NotImplementedError("We do not support the task to perturb messages: alfworld")

def perturb_messages(instruction, messages, task_name="webshop"):
    if task_name == 'webshop':
        return perturb_messages_for_webshop(instruction, messages)
    elif task_name == 'sciworld':
        return perturb_messages_for_sciworld(instruction, messages)
    elif task_name == 'alfworld':
        return perturb_messages_for_alfworld(instruction, messages)
    else:
        raise NotImplementedError(f"We do not support the task: {task_name}")

@torch.no_grad()
def evaluate_trajs_qnet_v2(model, tokenizer, new_state, batch_size=1, disable_tqdm=False, model_name='/mnt/model/Llama-2-7b-chat-hf/', debug=False):
    q_values = []
    # sources = [new_state.to_dict()['conversations'][:-1]]
    # if not disable_tqdm:
    #     progress = tqdm.tqdm(total=len(sources), desc="Evaluating Trajectories")
    conv = new_state.to_dict()['conversations']
    # We want the last message to be the GPT action (no following observation)
    if conv and conv[-1].get("from") == "human":
        conv = conv[:-1]
    sources = [conv]
    device = next(model.parameters()).device
    
    from qlass.data_utils import preprocess
    # data_dict = preprocess(sources, tokenizer, model_name, [0. for i in range(len(sources))])
    # -------- QNet-safe trimming to avoid losing the tail via HF truncation --------
    # Keep first 3 messages, then drop oldest turns (2 msgs) until prompt fits.
    from qlass.data_utils import get_chat_template
    chat = get_chat_template(model_name)

    def _prompt_tokens(convs):
        # convert conversations -> messages for prompt building
        roles = {"human": "user", "gpt": "assistant"}
        roles_list = ["user", "assistant"]
        src = convs
        if roles[src[0]["from"]] != "user":
            src = src[1:]
        msgs = []
        for j, s in enumerate(src):
            role = roles[s["from"]]
            assert role == roles_list[j % 2]
            msgs.append({"role": role, "content": s["value"]})
        prompt = chat.get_prompt(msgs + [{"role": "assistant", "content": None}])
        return len(tokenizer(prompt, add_special_tokens=True).input_ids)

    max_prompt_tokens = getattr(tokenizer, "model_max_length", 4096)
    # Use CLI caps if present (added below), else defaults
    max_prompt_tokens = min(max_prompt_tokens, getattr(model, "_qnet_max_prompt_tokens", 3800))
    keep_first_n = getattr(model, "_qnet_keep_first_n", 3)
    min_tail_msgs = getattr(model, "_qnet_min_tail_msgs", 4)

    trimmed = sources[0]
    n_tok = _prompt_tokens(trimmed)
    if n_tok > max_prompt_tokens and len(trimmed) > keep_first_n + min_tail_msgs:
        prefix = trimmed[:keep_first_n]
        rest = trimmed[keep_first_n:]
        # remove oldest turns by 2 messages to preserve alternation
        while n_tok > max_prompt_tokens and len(rest) > min_tail_msgs:
            if len(rest) >= 2:
                rest = rest[2:]
            else:
                break
            trimmed = prefix + rest
            n_tok = _prompt_tokens(trimmed)
        sources = [trimmed]
    # ---------------------------------------------------------------------------

    # data_dict = preprocess(sources, tokenizer, model_name, [0.0 for _ in range(len(sources))])
    
    # for i in range(0, len(sources), batch_size):
    #     batch_input_ids = data_dict['input_ids'][i:i + batch_size].to(device)
    #     attention_mask = data_dict['attention_mask'][i:i + batch_size].to(device)
    #     # QNet forward pass
    #     q_output = model(batch_input_ids, attention_mask=attention_mask)
    #     # Assuming QNet returns a single scalar per example as Q-value
    #     # import pdb; pdb.set_trace()
    #     batch_q_values = q_output[:,-1].squeeze().tolist()  # Ensure it's a list
    #     if isinstance(batch_q_values, float):
    #         batch_q_values = [batch_q_values]

    #     q_values.extend(batch_q_values)

    # ---- Force left truncation as a safety net (keeps tail, avoids >4096) ----
    old_side = getattr(tokenizer, "truncation_side", "right")
    old_max_len = getattr(tokenizer, "model_max_length", 4096)
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = max_prompt_tokens

    data_dict = preprocess(sources, tokenizer, model_name, [0.0 for _ in range(len(sources))])

    # restore tokenizer settings
    tokenizer.truncation_side = old_side
    tokenizer.model_max_length = old_max_len
    # ------------------------------------------------------------------------

    for i in range(0, len(sources), batch_size):
        batch_input_ids = data_dict['input_ids'][i:i + batch_size].to(device)
        attention_mask = data_dict['attention_mask'][i:i + batch_size].to(device)

        q_output = model(batch_input_ids, attention_mask=attention_mask)

        # unwrap outputs if needed
        if isinstance(q_output, (tuple, list)):
            q_output = q_output[0]

        # normalize shapes to [B, T] when possible
        if hasattr(q_output, "dim") and q_output.dim() == 3 and q_output.size(-1) == 1:
            q_output = q_output.squeeze(-1)  # [B, T]

        # Now extract per-example scalar Q safely
        if q_output.dim() == 2:
            # last non-pad index (safe)
            lengths = attention_mask.long().sum(dim=1)  # [B]
            lengths = torch.clamp(lengths, min=1)       # avoid 0 -> last_idx=-1
            last_idx = lengths - 1
            last_idx = torch.clamp(last_idx, min=0, max=q_output.size(1) - 1)

            batch_q_values = q_output[torch.arange(q_output.size(0), device=q_output.device), last_idx]
        elif q_output.dim() == 1:
            # already [B]
            batch_q_values = q_output
        else:
            # very defensive fallback
            batch_q_values = q_output.reshape(q_output.size(0), -1)[:, 0]

        batch_q_values = batch_q_values.detach().float().cpu().tolist()
        if isinstance(batch_q_values, float):
            batch_q_values = [batch_q_values]

        q_values.extend(batch_q_values)
    
    if debug:
        print(sources[0])
        print(q_values[0])

    return q_values

# need to deploy at local env.
def draw_graph_v2(G, pos, idx=0):
    # Use Graphviz layout to draw the tree
    pos = graphviz_layout(G, prog='dot')  # 'dot' engine gives a top-down layout
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=9, edge_color='gray', font_weight='bold')
    plt.savefig(f"tree_{idx}.png")
    plt.show()
    
def draw_graph(G, pos, idx=0):
    labels = {node: G.nodes[node]['label'] for node in G}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2500, node_color='skyblue', font_size=9, font_weight='bold', edge_color='gray')
    plt.savefig(f"tree_{idx}.png")
    plt.show()
    

def collect_and_print_stats(node):
    """ Recursively collect stats from the tree and print them. """
    def recurse_collect(node):
        rewards = [node.reward] if node.reward is not None else []
        q_values = [node.q_value] if node.q_value is not None else []
        
        for child in node.children:
            child_rewards, child_q_values = recurse_collect(child)
            rewards.extend(child_rewards)
            q_values.extend(child_q_values)
        return rewards, q_values
    
    rewards, q_values = recurse_collect(node)
    
    if rewards:
        average_reward = sum(rewards) / len(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
    else:
        average_reward = min_reward = max_reward = None

    if q_values:
        average_q_value = sum(q_values) / len(q_values)
        min_q_value = min(q_values)
        max_q_value = max(q_values)
    else:
        average_q_value = min_q_value = max_q_value = None

    print(f"Average Reward: {average_reward}")
    print(f"Min Reward: {min_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Average Q Value: {average_q_value}")
    print(f"Min Q Value: {min_q_value}")
    print(f"Max Q Value: {max_q_value}")

def find_top_k_trajectories(root, k=3):
    """ This function finds the top k trajectories based on the sum of Q-values. """
    # Store the sum of Q-values for each trajectory
    trajectory_sums = {}

    def calculate_q_sum(node, current_path, current_sum):
        current_sum += node.q_value
        if not node.children:
            # If no children, it means the path ends here, record the sum
            trajectory_sums[tuple(current_path)] = current_sum
        else:
            # Otherwise, continue down each child
            for child in node.children:
                calculate_q_sum(child, current_path + [child], current_sum)

    # Start the recursive function from root
    calculate_q_sum(root, [root], 0)

    # Sort the trajectories by their Q-value sums, in descending order
    sorted_trajectories = sorted(trajectory_sums.items(), key=lambda item: item[1], reverse=True)

    # Return the top k trajectories
    return sorted_trajectories[:k]

def verify_conversations(conversations):
    if not conversations:
        return False  # Return False if the list is empty
    
    # Check if the list starts with a human message and alternates properly
    if conversations[0]['from'] != 'human':
        return False  # First message must be from the human
    
    # Verify alternating roles and that it ends with gpt
    expected_from = 'human'
    for message in conversations:
        if message['from'] != expected_from:
            return False  # 'from' field does not match the expected role
        # Switch the expected role for the next iteration
        expected_from = 'gpt' if expected_from == 'human' else 'human'
    
    # Check if the last message is from gpt
    if conversations[-1]['from'] != 'user':
        return False  # Last message must be from gpt
    
    return True  # All checks passed


# GEN_FUNC = interactive_loop_for_explore
# RECURSIVE_MODE = True
# N_SFT_EXAMPLES = 1000

MAX_TURNS={"webshop":5,"sciworld":40,"alfworld":40}
N_SAMPLE = 3
# N_TRAJS = 3
EPSILON = 0.1
TOPK = 2


def _force_min_max_steps(env, min_steps: int) -> int:
    """SciWorld reset can overwrite env.max_steps (e.g. from max_steps.json). Force it back."""
    cur = getattr(env, "max_steps", None)
    if cur is None:
        env.max_steps = int(min_steps)
    else:
        env.max_steps = int(max(int(cur), int(min_steps)))
    return int(env.max_steps)


def is_q_adv_logit_strategy(args) -> bool:
    return args.selection_strategy in {"q_adv_logit_argmax", "q_adv_logit_sample"}


def uses_canonical_oversampled_generation(args) -> bool:
    """Use the same candidate generator as q_adv_logit, but not necessarily the same selector.

    q_argmax_oversample is the compute-matched raw-Q baseline:
      - generate candidates with canonical prompt + oversampling + dedup;
      - request actor logprobs for compute parity / diagnostics;
      - ignore logprobs at selection time;
      - select argmax(selection_score_raw).
    """
    return args.selection_strategy in {
        "q_adv_logit_argmax",
        "q_adv_logit_sample",
        "q_argmax_oversample",
    }


def get_candidate_raw_selection_score(candidate: Dict[str, Any]) -> float:
    """Score used by the current QLASS selection logic.

    For unfinished candidates this is QNet score.
    For finished candidates this is environment reward.
    """
    return float(candidate["selection_score_raw"])


def get_candidate_actor_logprob(candidate: Dict[str, Any], args) -> float:
    """Return action-level actor logprob used in corrected-score selection.

    Fail loudly if logprobs are missing in q_adv_logit mode, because without
    them the method is no longer actor-logit correction.
    """
    if args.actor_logprob_type == "mean":
        value = candidate.get("actor_logprob_mean")
    elif args.actor_logprob_type == "sum":
        value = candidate.get("actor_logprob_sum")
    else:
        raise ValueError(f"Unsupported actor_logprob_type: {args.actor_logprob_type}")

    if value is None:
        raise RuntimeError(
            "[q_guided_inference] Missing actor logprob for q_adv_logit selection. "
            f"idx={candidate.get('idx')} action={candidate.get('action')!r} "
            f"actor_logprob_type={args.actor_logprob_type}"
        )

    return float(value)


def compute_q_advantage_and_corrected_scores(
    candidate_records: List[Dict[str, Any]],
    args,
) -> None:
    """Fill q_mean/q_std/q_advantage/corrected_score in-place.

    Q-like signal:
      - for unfinished candidates: QNet score;
      - for finished candidates: env reward;
    both are already stored in selection_score_raw.

    Corrected score:
        corrected_score = actor_logprob_coef * actor_logprob
                        + q_adv_beta * q_advantage
    """
    if not candidate_records:
        raise RuntimeError("[q_guided_inference] Cannot compute scores for empty candidates.")

    q_values = np.array(
        [float(candidate["selection_score_raw"]) for candidate in candidate_records],
        dtype=np.float64,
    )

    q_mean = float(np.mean(q_values))
    q_std = float(np.std(q_values))

    if len(candidate_records) < 2 or q_std <= args.q_adv_eps:
        logger.warning(
            "[Q_ADV] Degenerate Q-score distribution: num_candidates=%d q_mean=%.6f "
            "q_std=%.6f eps=%.6f. Setting q_advantage=0 for all candidates.",
            len(candidate_records),
            q_mean,
            q_std,
            args.q_adv_eps,
        )
        q_advantages = np.zeros_like(q_values, dtype=np.float64)
    else:
        q_advantages = (q_values - q_mean) / (q_std + args.q_adv_eps)
        q_advantages = np.clip(q_advantages, -args.q_adv_clip, args.q_adv_clip)

    for candidate, q_advantage in zip(candidate_records, q_advantages.tolist()):
        actor_logprob = get_candidate_actor_logprob(candidate, args)

        corrected_score = (
            args.actor_logprob_coef * actor_logprob
            + args.q_adv_beta * float(q_advantage)
        )

        candidate["q_mean"] = q_mean
        candidate["q_std"] = q_std
        candidate["q_advantage"] = float(q_advantage)
        candidate["actor_logprob_used"] = float(actor_logprob)
        candidate["corrected_score"] = float(corrected_score)

    # logger.info(
    #     "[Q_ADV] Computed corrected scores: num_candidates=%d q_mean=%.6f q_std=%.6f "
    #     "actor_logprob_type=%s actor_coef=%.6f q_adv_beta=%.6f q_adv_clip=%.6f",
    #     len(candidate_records),
    #     q_mean,
    #     q_std,
    #     args.actor_logprob_type,
    #     args.actor_logprob_coef,
    #     args.q_adv_beta,
    #     args.q_adv_clip,
    # )


def get_candidate_corrected_score(candidate: Dict[str, Any]) -> float:
    value = candidate.get("corrected_score")
    if value is None:
        raise RuntimeError(
            "[q_guided_inference] corrected_score is missing. "
            f"idx={candidate.get('idx')} action={candidate.get('action')!r}"
        )
    return float(value)


def select_candidate_q_adv_logit(
    candidate_records: List[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    """Select candidate using corrected scores.

    q_adv_logit_argmax:
        deterministic argmax over corrected_score.

    q_adv_logit_sample:
        sample from softmax(corrected_score / temperature).
    """
    if not candidate_records:
        raise RuntimeError("[q_guided_inference] Cannot select from empty candidate_records.")

    if args.selection_strategy == "q_adv_logit_argmax":
        selected = max(candidate_records, key=get_candidate_corrected_score)
        logger.info(
            "[SELECT_CORRECTED] argmax selected idx=%s action=%r corrected=%.6f "
            "raw=%.6f q_adv=%.6f actor_logprob=%s",
            selected.get("idx"),
            selected.get("action"),
            get_candidate_corrected_score(selected),
            get_candidate_raw_selection_score(selected),
            float(selected.get("q_advantage", 0.0)),
            selected.get("actor_logprob_used"),
        )
        return selected

    if args.selection_strategy == "q_adv_logit_sample":
        scores = np.array(
            [get_candidate_corrected_score(candidate) for candidate in candidate_records],
            dtype=np.float64,
        )

        temperature = float(args.q_adv_sample_temperature)
        if temperature <= 0:
            raise ValueError(
                f"--q_adv_sample_temperature must be positive, got {temperature}"
            )

        scaled_scores = scores / temperature
        scaled_scores = scaled_scores - np.max(scaled_scores)
        probs = np.exp(scaled_scores)
        probs_sum = float(np.sum(probs))

        if not np.isfinite(probs_sum) or probs_sum <= 0:
            raise RuntimeError(
                "[q_guided_inference] Invalid softmax probabilities for q_adv_logit_sample: "
                f"scores={scores.tolist()} temperature={temperature}"
            )

        probs = probs / probs_sum
        selected_idx = int(np.random.choice(len(candidate_records), p=probs))
        selected = candidate_records[selected_idx]

        logger.info(
            "[SELECT_CORRECTED] sample selected local_idx=%d idx=%s action=%r "
            "corrected=%.6f prob=%.6f temperature=%.6f raw=%.6f q_adv=%.6f "
            "actor_logprob=%s all_probs=%s all_scores=%s",
            selected_idx,
            selected.get("idx"),
            selected.get("action"),
            get_candidate_corrected_score(selected),
            float(probs[selected_idx]),
            temperature,
            get_candidate_raw_selection_score(selected),
            float(selected.get("q_advantage", 0.0)),
            selected.get("actor_logprob_used"),
            [round(float(p), 6) for p in probs.tolist()],
            [round(float(s), 6) for s in scores.tolist()],
        )
        return selected

    raise NotImplementedError(
        f"select_candidate_q_adv_logit does not support selection_strategy={args.selection_strategy}"
    )


def compute_and_mark_selection_diagnostics(
    candidate_records: List[Dict[str, Any]],
    selected_candidate: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    """Compute per-step diagnostics comparing raw-Q and corrected-score choices.

    Marks candidate records in-place with:
      - selected_by_raw_q
      - selected_by_corrected_argmax
      - selected_by_final_policy

    Definitions:
      raw_best:
        candidate selected by original QLASS raw score.

      corrected_argmax:
        candidate with max corrected_score.
        Defined only for q_adv_logit strategies.

      selected_candidate:
        actually selected candidate.
        For q_adv_logit_argmax: equals corrected_argmax.
        For q_adv_logit_sample: sampled from corrected-score distribution.
    """
    if not candidate_records:
        raise RuntimeError("[q_guided_inference] Empty candidate_records in diagnostics.")

    raw_best = max(candidate_records, key=get_candidate_raw_selection_score)

    if is_q_adv_logit_strategy(args):
        corrected_argmax = max(candidate_records, key=get_candidate_corrected_score)
    else:
        corrected_argmax = None

    raw_best_action = raw_best.get("action")
    selected_action = selected_candidate.get("action")

    raw_vs_selected_changed = actions_differ_for_change_stats(
        raw_best_action,
        selected_action,
    )

    raw_best_idx = raw_best.get("idx")
    corrected_argmax_idx = corrected_argmax.get("idx") if corrected_argmax is not None else None
    selected_idx = selected_candidate.get("idx")

    for candidate in candidate_records:
        candidate["selected_by_raw_q"] = candidate.get("idx") == raw_best_idx
        candidate["selected_by_corrected_argmax"] = (
            corrected_argmax is not None and candidate.get("idx") == corrected_argmax_idx
        )
        candidate["selected_by_final_policy"] = candidate.get("idx") == selected_idx

    diagnostics = {
        "is_q_adv_logit_strategy": bool(is_q_adv_logit_strategy(args)),
        "selection_strategy": args.selection_strategy,

        "raw_best_idx": raw_best_idx,
        "raw_best_action": raw_best_action,
        "raw_best_score": get_candidate_raw_selection_score(raw_best),

        "selected_idx": selected_idx,
        "selected_action": selected_action,
        "selected_corrected_score": selected_candidate.get("corrected_score"),

        "raw_vs_selected_changed": bool(raw_vs_selected_changed),
    }

    logger.info(
        "[STEP_CORRECTION_DIAG] strategy=%s raw_best_idx=%s raw_best_action=%r "
        "selected_idx=%s "
        "selected_action=%r "
        "raw_vs_selected_changed=%s",
        diagnostics["selection_strategy"],
        diagnostics["raw_best_idx"],
        diagnostics["raw_best_action"],
        diagnostics["selected_idx"],
        diagnostics["selected_action"],
        diagnostics["raw_vs_selected_changed"],
    )

    return diagnostics


def sync_action_value_dict_with_candidate_records(
    action_value_dict: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
) -> None:
    """Copy computed q-adv/corrected fields from candidate records to JSON logs."""
    by_idx = {candidate["idx"]: candidate for candidate in candidate_records}

    for item in action_value_dict:
        candidate = by_idx.get(item["idx"])
        if candidate is None:
            raise RuntimeError(
                "[q_guided_inference] action_value_dict contains idx not found in "
                f"candidate_records: idx={item['idx']}"
            )

        item["q_mean"] = candidate.get("q_mean")
        item["q_std"] = candidate.get("q_std")
        item["q_advantage"] = candidate.get("q_advantage")
        item["actor_logprob_used"] = candidate.get("actor_logprob_used")
        item["corrected_score"] = candidate.get("corrected_score")
        item["selected"] = bool(candidate.get("selected", False))
        item["selected_by_raw_q"] = bool(candidate.get("selected_by_raw_q", False))
        item["selected_by_final_policy"] = bool(
            candidate.get("selected_by_final_policy", False)
        )


def select_candidate_raw_q_argmax(
    candidate_records: List[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    """Select candidate using the original QLASS logic.

    This function intentionally ignores actor logprobs and future corrected scores.
    It is used for the infrastructure step where candidate records and logprobs
    are added, but the final selection rule is not changed yet.
    """
    if not candidate_records:
        raise RuntimeError("[q_guided_inference] Cannot select from empty candidate_records.")

    if args.sample_mode == "epsilon_greedy":
        if random.random() < EPSILON:
            selected = random.choice(candidate_records)
            logger.info(
                "[SELECT_RAW] epsilon_greedy explored: selected idx=%s action=%r raw_score=%.6f",
                selected.get("idx"),
                selected.get("action"),
                get_candidate_raw_selection_score(selected),
            )
            return selected

        selected = max(candidate_records, key=get_candidate_raw_selection_score)
        logger.info(
            "[SELECT_RAW] epsilon_greedy exploited: selected idx=%s action=%r raw_score=%.6f",
            selected.get("idx"),
            selected.get("action"),
            get_candidate_raw_selection_score(selected),
        )
        return selected

    if args.sample_mode == "bon":
        selected = max(candidate_records, key=get_candidate_raw_selection_score)
        logger.info(
            "[SELECT_RAW] bon selected idx=%s action=%r raw_score=%.6f",
            selected.get("idx"),
            selected.get("action"),
            get_candidate_raw_selection_score(selected),
        )
        return selected

    raise NotImplementedError(f"We do not support the sample mode: {args.sample_mode}")


def log_candidate_records(
    candidate_records: List[Dict[str, Any]],
    *,
    task_id: Any,
    traj_id: int,
    n_turn: int,
    selection_strategy: str,
) -> None:
    # logger.info(
    #     "[CANDIDATES] task=%s traj=%d turn=%d strategy=%s num_candidates=%d",
    #     task_id,
    #     traj_id,
    #     n_turn,
    #     selection_strategy,
    #     len(candidate_records),
    # )

    for cand in candidate_records:
        logger.info(
            "[CANDIDATE] task=%s traj=%d turn=%d idx=%s action=%r "
            "finished=%s env_reward=%s raw_score=%.6f q_score=%s "
            "q_mean=%s q_std=%s q_advantage=%s corrected_score=%s "
            "actor_logprob_used=%s actor_logprob_mean=%s actor_logprob_sum=%s "
            "num_tokens=%s selected=%s selected_by_raw_q=%s selected_by_final_policy=%s generation_mode=%s raw_index=%s round_index=%s",
            task_id,
            traj_id,
            n_turn,
            cand.get("idx"),
            cand.get("action"),
            cand.get("finished"),
            cand.get("env_reward"),
            float(cand.get("selection_score_raw", 0.0)),
            cand.get("q_score"),
            cand.get("q_mean"),
            cand.get("q_std"),
            cand.get("q_advantage"),
            cand.get("corrected_score"),
            cand.get("actor_logprob_used"),
            cand.get("actor_logprob_mean"),
            cand.get("actor_logprob_sum"),
            cand.get("num_output_tokens"),
            cand.get("selected"),
            cand.get("selected_by_raw_q"),
            cand.get("selected_by_final_policy"),
            cand.get("generation_mode"),
            cand.get("raw_index"),
            cand.get("round_index"),
        )


def safe_float_or_none(value: Any) -> float | None:
    """Convert numeric value to float, keep None as None.

    Some environments/states can have reward=None for non-terminal intermediate
    states. This should not break candidate logging, because raw selection score
    for unfinished states is QNet score.
    """
    if value is None:
        return None
    return float(value)


def reward_to_float(value: Any) -> float:
    """Convert trajectory reward to metric-safe float.

    In ALFWorld/SciWorld unfinished trajectories can have reward=None.
    For evaluation, None should be treated as 0.0.
    """
    if value is None:
        return 0.0
    return float(value)


def normalize_action_for_change_stats(action: Any) -> str:
    """Normalize action string for correction-change statistics."""
    if action is None:
        return ""
    return " ".join(str(action).strip().lower().split())


def actions_differ_for_change_stats(action_a: Any, action_b: Any) -> bool:
    """Return True if two actions should be treated as different."""
    return normalize_action_for_change_stats(action_a) != normalize_action_for_change_stats(action_b)


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def main(args):
    if is_q_adv_logit_strategy(args) and args.disable_perturb is False:
        logger.info(
            "[q_guided_inference] q_adv_logit strategy uses canonical single-prompt candidate generation with logprobs. The old explore_add_prompt perturbation "
            "will be ignored for candidate generation in this mode."
        )

    total_examples = []
    total_trees = []
    successful_trajs = []
    token_count = 0

    global_correction_stats = {
        "num_q_adv_steps": 0,
        "num_raw_vs_selected_changed": 0,
    }
    
    args.model_name_or_path = args.qnet_path
    args.low_cpu_mem_usage = False
    args.use_flash_attn = True
    qnet = QNet.from_pretrained(args.qnet_path,None,args)
    qnet = qnet.to("cuda")
    qnet.device = torch.device("cuda")

    qnet.mode = "final"
    qnet.eval()

    print("loaded qnet successfully")
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
    #                                             model_max_length=4096,
    #                                             use_fast=False)
    # Tokenizer MUST match the SFT model (same chat template/tokenizer as action generator)
    from transformers import AutoTokenizer
    # Resolve tokenizer path: explicit arg > agent_config tokenizer_path > args.model_name
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    tokenizer_path = args.tokenizer_path or agent_config.get("config", {}).get("tokenizer_path") or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=4096, use_fast=False)
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    random.seed(42)

    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    # with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
    #     agent_config: Dict[str, Any] = json.load(f)
        
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name
        agent_config['config']['batch_size'] = args.eval_batch_size

    # Attach QNet trimming caps to the model instance (used inside evaluate_trajs_qnet_v2)
    qnet._qnet_max_prompt_tokens = args.qnet_max_prompt_tokens
    qnet._qnet_keep_first_n = args.qnet_keep_first_n
    qnet._qnet_min_tail_msgs = args.qnet_min_tail_msgs
        
    env_config = exp_config["env_config"]
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")
    
    if env_config['env_class'] == 'WebShopEnv':
        from eval.webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.slice_num, args.slice_id) 
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    traj_file = args.output_dir+f"{args.slice_id}of{args.slice_num}_slices_{args.sample_mode}_traj.jsonl"
    tree_file =args.output_dir+f"{args.slice_id}of{args.slice_num}_slices_{args.sample_mode}_tree.pkl"
    
    turn_cap = MAX_TURNS[args.exp_config]   # for sciworld => 40

    done_task_id = []
    mult_success_num = 0
    num_first_success = 0
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)
        for i, task in enumerate(all_tasks):
            if args.debug and i==2:
                break
            
            ds = args.exp_config
            all_trajs = []
            print(f"Task {i}")
            
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue
            
            env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)

            print(
                f"[DBG_MAX_STEPS] BEFORE override: "
                f"args.exp_config={args.exp_config} "
                f"env_config.max_steps={env_config.get('max_steps', None)} "
                f"env.max_steps={getattr(env, 'max_steps', None)} "
                f"MAX_TURNS={MAX_TURNS.get(args.exp_config, None)}"
            )

            env.max_steps = _force_min_max_steps(env, turn_cap)

            print(f"[DBG_MAX_STEPS] AFTER override: env.max_steps={getattr(env, 'max_steps', None)}")

            if args.force_first:
                env.icl_format = 'first'
            
            if env.icl_format == 'first':
                start_i = 1
            elif env.icl_format == 'conversation':
                start_i = 2

            init_msg, state = env.reset(num_icl_examples=args.num_icl_examples)
            _force_min_max_steps(env, turn_cap)
            print(f"[DBG_MAX_STEPS] AFTER reset#0 (forced): env.max_steps={getattr(env,'max_steps',None)}")

            root = TreeNode(state=init_msg, action='No Action (Root)', reward=state.reward) 
            
            instruction_path = f"{root_dir}/eval_agent/prompt/instructions/{args.exp_config}_inst.txt"
            with open(instruction_path) as f:
                instruction = f.read()
            num_success = 0 
            for traj_id in range(args.n_trajs):
                init_msg , cur_traj_state = env.reset(num_icl_examples=args.num_icl_examples)

                _force_min_max_steps(env, turn_cap)
                print(f"[DBG_MAX_STEPS] AFTER reset(traj{traj_id}) (forced): env.max_steps={getattr(env,'max_steps',None)}")

                action_value_list = []

                step_selection_diagnostics_list = []

                traj_correction_stats = {
                    "num_q_adv_steps": 0,
                    "num_raw_vs_selected_changed": 0,
                    "raw_vs_selected_changed_rate": 0.0,
                }
                # for n_turn in range(turn_cap):
                for n_turn in range(env.max_steps):
                    candidate_records: List[Dict[str, Any]] = []

                    if n_turn == 0:
                        current_node = root

                    action_value_dict: List[Dict[str, Any]] = []

                    def extract_assistant_action_msgs(history):
                        out = []
                        for msg in history:
                            if msg["role"] != "assistant":
                                continue
                            text = msg["content"].strip()
                            # Keep exactly the assistant messages that were actually sent to env.step:
                            # either normal Thought/Action outputs or numeric disambiguation replies.
                            if ("Action:" in text) or text.isdigit():
                                out.append(text)
                        return out

                    # ------------------------------------------------------------------
                    # Candidate generation.
                    # Old QLASS path:
                    #   - args.best_of_N separate calls;
                    #   - idx > 0 may use explore_add_prompt / perturbation.
                    #
                    # New q_adv_logit path:
                    #   - one canonical prompt;
                    #   - oversampling + dedup inside SGLangAgent;
                    #   - each candidate has actor output-token logprobs.
                    # ------------------------------------------------------------------
                    if uses_canonical_oversampled_generation(args):
                        generated_candidates = agent.generate_candidates_with_logprobs(
                            cur_traj_state.history,
                            n=args.best_of_N,
                            add_action_candidate_prompt=args.q_adv_add_candidate_suffix,
                            oversample_factor=args.candidate_oversample_factor,
                            max_rounds=args.candidate_max_rounds,
                        )

                        logger.info(
                            "[GEN_CANONICAL_OVERSAMPLE] task=%s traj=%d turn=%d strategy=%s "
                            "requested_best_of_N=%d received_unique_candidates=%d",
                            task.task_id,
                            traj_id,
                            n_turn,
                            args.selection_strategy,
                            args.best_of_N,
                            len(generated_candidates),
                        )

                        candidate_generation_items = []
                        for idx, cand in enumerate(generated_candidates):
                            candidate_generation_items.append(
                                {
                                    "idx": idx,
                                    "action_text_for_env": cand["text"],
                                    "parsed_action": cand.get("action"),
                                    "actor_logprob_sum": cand.get("actor_logprob_sum"),
                                    "actor_logprob_mean": cand.get("actor_logprob_mean"),
                                    "num_output_tokens": cand.get("num_output_tokens"),
                                    "finish_reason": cand.get("finish_reason"),
                                    "raw_index": cand.get("raw_index"),
                                    "round_index": cand.get("round_index"),
                                    "generation_mode": "canonical_oversample",
                                }
                            )
                    else:
                        candidate_generation_items = []

                        action_list = []
                        for idx in range(args.best_of_N):
                            if (not idx) or args.disable_perturb:
                                cur_state_history = cur_traj_state.history
                            else:
                                assert not args.disable_perturb
                                if args.exp_config == "webshop":
                                    cur_state_history = perturb_messages(
                                        instruction,
                                        copy.deepcopy(cur_traj_state.history),
                                    )
                                else:
                                    # Old QLASS diversity prompt. This path is intentionally
                                    # not used by q_adv_logit strategies because it changes the
                                    # policy context and makes actor logprobs incomparable.
                                    if idx == 1:
                                        explore_add_prompt = (
                                            f"\nPlease provide another reasonable response "
                                            f"different from '{action_list[0]}'."
                                        )
                                    else:
                                        explore_add_prompt = "\nYou have given the the following answers.\n"
                                        for prev_i in range(len(action_list)):
                                            explore_add_prompt += (
                                                f"Answer {prev_i + 1}: '{action_list[prev_i]}'\n"
                                            )
                                        explore_add_prompt += (
                                            "Please provide a reasonable response different "
                                            "from previous answers."
                                        )

                                    cur_state_history = copy.deepcopy(cur_traj_state.history)
                                    cur_state_history[-1]["content"] += explore_add_prompt

                            action = agent(cur_state_history)
                            action_list.append(action)

                            candidate_generation_items.append(
                                {
                                    "idx": idx,
                                    "action_text_for_env": action,
                                    "parsed_action": None,
                                    "actor_logprob_sum": None,
                                    "actor_logprob_mean": None,
                                    "num_output_tokens": None,
                                    "finish_reason": None,
                                    "raw_index": idx,
                                    "round_index": 0,
                                    "generation_mode": "old_q_guided",
                                }
                            )

                    # ------------------------------------------------------------------
                    # Candidate evaluation.
                    # Evaluate all generated candidates with the same replay + env.step +
                    # QNet logic as the original QLASS implementation.
                    # ------------------------------------------------------------------
                    for cand_item in candidate_generation_items:
                        idx = int(cand_item["idx"])
                        action = cand_item["action_text_for_env"]

                        observation, state = env.reset(num_icl_examples=args.num_icl_examples)
                        _force_min_max_steps(env, turn_cap)

                        if n_turn > 0:
                            prefix_actions = extract_assistant_action_msgs(cur_traj_state.history)[:n_turn]
                            for act_text in prefix_actions:
                                observation, state = env.step(act_text)

                            replay_actions = extract_assistant_action_msgs(state.history)
                            traj_actions = extract_assistant_action_msgs(cur_traj_state.history)
                            assert replay_actions == traj_actions, (
                                f"[REPLAY ACTION MISMATCH] n_turn={n_turn}\n"
                                f"replay_actions={replay_actions}\n"
                                f"traj_actions={traj_actions}\n"
                            )

                        observation, new_state = env.step(action)

                        env_reward = safe_float_or_none(new_state.reward)

                        if new_state.finished:
                            if env_reward is None:
                                raise RuntimeError(
                                    "[q_guided_inference] Finished candidate has reward=None. "
                                    f"task={task.task_id} traj={traj_id} turn={n_turn} "
                                    f"idx={idx} action={action!r}"
                                )

                            q_score = None
                            selection_score_raw = float(env_reward)
                        else:
                            q_score = float(
                                evaluate_trajs_qnet_v2(
                                    qnet,
                                    tokenizer,
                                    new_state,
                                    batch_size=1,
                                    disable_tqdm=True,
                                    model_name=args.model_name,
                                    debug=args.debug,
                                )[0]
                            )
                            selection_score_raw = q_score

                        parsed_action = cand_item.get("parsed_action")
                        if parsed_action is None:
                            parsed_action = action

                        candidate_record = {
                            "idx": idx,
                            "action": parsed_action,
                            "action_text_for_env": action,
                            "new_state": new_state,
                            "finished": bool(new_state.finished),
                            "env_reward": env_reward,
                            "q_score": q_score,
                            "selection_score_raw": float(selection_score_raw),
                            # The same scalar is used as the Q-like signal for z-score advantage.
                            "q_for_advantage": float(selection_score_raw),
                            "actor_logprob_sum": cand_item.get("actor_logprob_sum"),
                            "actor_logprob_mean": cand_item.get("actor_logprob_mean"),
                            "actor_logprob_used": None,
                            "num_output_tokens": cand_item.get("num_output_tokens"),
                            "finish_reason": cand_item.get("finish_reason"),
                            "raw_index": cand_item.get("raw_index"),
                            "round_index": cand_item.get("round_index"),
                            "generation_mode": cand_item.get("generation_mode"),
                            # Filled in the next implementation step.
                            "q_mean": None,
                            "q_std": None,
                            "q_advantage": None,
                            "corrected_score": None,
                            "selected": False,
                            "selected_by_raw_q": False,
                            "selected_by_final_policy": False,
                        }

                        candidate_records.append(candidate_record)

                        action_value_dict.append(
                            {
                                "idx": idx,
                                "action": candidate_record["action_text_for_env"],
                                "parsed_action": candidate_record["action"],
                                "value": candidate_record["selection_score_raw"],
                                "q_score": candidate_record["q_score"],
                                "selection_score_raw": candidate_record["selection_score_raw"],
                                "q_for_advantage": candidate_record["q_for_advantage"],
                                "actor_logprob_sum": candidate_record["actor_logprob_sum"],
                                "actor_logprob_mean": candidate_record["actor_logprob_mean"],
                                "actor_logprob_used": candidate_record["actor_logprob_used"],
                                "num_output_tokens": candidate_record["num_output_tokens"],
                                "finished": candidate_record["finished"],
                                "env_reward": candidate_record["env_reward"],
                                "generation_mode": candidate_record["generation_mode"],
                                "raw_index": candidate_record["raw_index"],
                                "round_index": candidate_record["round_index"],
                                "q_mean": None,
                                "q_std": None,
                                "q_advantage": None,
                                "corrected_score": None,
                                "selected": False,
                                "selected_by_raw_q": False,
                                "selected_by_final_policy": False,
                            }
                        )

                        # if i < 2 and n_turn < 3:
                        #     score_for_print = candidate_record["selection_score_raw"]
                        #     print(
                        #         f"[DBG] idx={idx} raw_score={score_for_print:.4f} "
                        #         f"q={candidate_record['q_score']} "
                        #         f"logp_mean={candidate_record['actor_logprob_mean']} "
                        #         f"action={str(action).splitlines()[-1][:120]}"
                        #     )

                    if not candidate_records:
                        raise RuntimeError(
                            f"[q_guided_inference] No candidates generated/evaluated for "
                            f"task={task.task_id}, traj={traj_id}, turn={n_turn}."
                        )

                    # ------------------------------------------------------------------
                    # Compute corrected scores and select candidate.
                    # ------------------------------------------------------------------
                    if is_q_adv_logit_strategy(args):
                        compute_q_advantage_and_corrected_scores(candidate_records, args)
                        selected_candidate = select_candidate_q_adv_logit(candidate_records, args)
                    else:
                        selected_candidate = select_candidate_raw_q_argmax(candidate_records, args)

                    selected_candidate["selected"] = True

                    selection_diagnostics = compute_and_mark_selection_diagnostics(
                        candidate_records=candidate_records,
                        selected_candidate=selected_candidate,
                        args=args,
                    )
                    selection_diagnostics["turn"] = n_turn
                    selection_diagnostics["task_id"] = task.task_id
                    selection_diagnostics["traj_id"] = traj_id
                    step_selection_diagnostics_list.append(selection_diagnostics)

                    if selection_diagnostics["is_q_adv_logit_strategy"]:
                        traj_correction_stats["num_q_adv_steps"] += 1
                        global_correction_stats["num_q_adv_steps"] += 1

                        if selection_diagnostics["raw_vs_selected_changed"]:
                            traj_correction_stats["num_raw_vs_selected_changed"] += 1
                            global_correction_stats["num_raw_vs_selected_changed"] += 1

                    # Sync computed fields and selected flag into JSON-serializable logs.
                    sync_action_value_dict_with_candidate_records(
                        action_value_dict,
                        candidate_records,
                    )

                    log_candidate_records(
                        candidate_records,
                        task_id=task.task_id,
                        traj_id=traj_id,
                        n_turn=n_turn,
                        selection_strategy=args.selection_strategy,
                    )

                    action_value_list.append(action_value_dict)

                    new_state = selected_candidate["new_state"]

                    # Keep TreeNode.q_value compatible with original QLASS:
                    # store raw QNet/env score, not corrected_score.
                    chosen_score = selected_candidate["selection_score_raw"]

                    logger.info(
                        "[SELECT_FINAL] task=%s traj=%d turn=%d strategy=%s "
                        "selected_idx=%s action=%r raw_score=%.6f corrected_score=%s "
                        "q_advantage=%s actor_logprob_used=%s",
                        task.task_id,
                        traj_id,
                        n_turn,
                        args.selection_strategy,
                        selected_candidate.get("idx"),
                        selected_candidate.get("action"),
                        float(selected_candidate.get("selection_score_raw")),
                        selected_candidate.get("corrected_score"),
                        selected_candidate.get("q_advantage"),
                        selected_candidate.get("actor_logprob_used"),
                    )

                    # chosen_score is still the original raw QLASS selection score:
                    # QNet score for unfinished candidates or env reward for finished candidates.
                    new_node = TreeNode(
                        state=new_state.to_dict()['conversations'][:-2],
                        action=new_state.to_dict()['conversations'][-2],
                        reward=new_state.reward,
                    )
                    new_node.q_value = float(chosen_score)
                    assert isinstance(new_state.to_dict()['conversations'][-2], dict) and new_state.to_dict()['conversations'][-2]['from']=='gpt'
                    current_node.add_child(new_node)
                    current_node = new_node
                    cur_traj_state = new_state
                    if cur_traj_state.finished:
                        print(
                            f"[DBG_DONE] finished at n_turn={n_turn+1} "
                            f"env.max_steps={getattr(env,'max_steps',None)} "
                            f"state.steps={getattr(cur_traj_state,'steps',None)} "
                            f"terminate_reason={getattr(cur_traj_state,'terminate_reason',None)}"
                        )
                        break
                traj_correction_stats["raw_vs_selected_changed_rate"] = safe_rate(
                    traj_correction_stats["num_raw_vs_selected_changed"],
                    traj_correction_stats["num_q_adv_steps"],
                )
                logger.info(
                    "[TRAJ_CORRECTION_STATS] task=%s traj=%d strategy=%s "
                    "q_adv_steps=%d raw_vs_selected_changed=%d rate=%.4f ",
                    task.task_id,
                    traj_id,
                    args.selection_strategy,
                    traj_correction_stats["num_q_adv_steps"],
                    traj_correction_stats["num_raw_vs_selected_changed"],
                    traj_correction_stats["raw_vs_selected_changed_rate"],
                )
                all_trajs.append(
                    {
                        'dataset': ds,
                        'id': task.task_id,
                        'conversations': cur_traj_state.to_dict()['conversations'],
                        'reward': reward_to_float(cur_traj_state.reward),
                        'success': bool(cur_traj_state.success),
                        'action_value_dict': action_value_list,
                        'selection_diagnostics': step_selection_diagnostics_list,
                        'q_adv_correction_stats': traj_correction_stats,
                    }
                )
                if cur_traj_state.success:
                    num_success += 1
                    if traj_id == 0:
                        num_first_success += 1
                

            if num_success > 1:
                mult_success_num += 1

            collect_and_print_stats(root)
            # max_reward_traj = max(all_trajs, key=lambda traj: traj['reward'])
            # for here, we only save the best trajectory for self-training, but this step needs to be modified for the DPO version to get preference-pair
            # total_examples += [max_reward_traj]
            total_examples += all_trajs
            # check whether the task has the attribute of 'session_id'
            if hasattr(task, 'session_id'):
                id = task.session_id
            else:
                # otherwise, set it to i
                id = i
            total_trees.append({'dataset':ds, 'id': id, 'tree': root})
            
            if id % 2 == 0:
                with open(traj_file, 'w') as f:
                    json.dump(total_examples, f)
                
                # save as pickle
                with open(tree_file, 'wb') as f:
                    pickle.dump(total_trees, f)
                    
            done_task_id.append(task.task_id) 
            pbar.update(1)
        pbar.close()

    n_traj = len(total_examples)
    rewards = [reward_to_float(traj.get("reward")) for traj in total_examples]
    n_success = sum(1.0 for reward in rewards if reward == 1.0)

    global_correction_stats["raw_vs_selected_changed_rate"] = safe_rate(
        global_correction_stats["num_raw_vs_selected_changed"],
        global_correction_stats["num_q_adv_steps"],
    )

    global_stats_msg = (
        "[GLOBAL_CORRECTION_STATS] "
        f"strategy={args.selection_strategy} "
        f"q_adv_steps={global_correction_stats['num_q_adv_steps']} "
        f"raw_vs_selected_changed={global_correction_stats['num_raw_vs_selected_changed']} "
        f"rate={global_correction_stats['raw_vs_selected_changed_rate']:.4f}"
    )

    logger.info(global_stats_msg)
    print(global_stats_msg)

    print(f"Finally, The Number of Successful Trajectories: {n_success} / {n_traj} ")
    print(f"Finally, The Number of Successful Trajectories with multi evaluation: {mult_success_num} / {n_tasks} ")
    print(f"Finally, The Number of Successful Trajectories with first inference: {num_first_success} / {n_tasks} ")
    print(f"Average Reward: {sum(rewards) / n_traj if n_traj > 0 else 0.0}")

    if args.debug:
        traj_file = f'{root_dir}/data/train/explore/debug2.jsonl'
        tree_file = f'{root_dir}/data/train/explore/debug2.pkl'
        
    with open(traj_file, 'w') as f:
        json.dump(total_examples, f)
            
    with open(tree_file, 'wb') as f:
        pickle.dump(total_trees, f)
    global_stats_file = os.path.join(args.output_dir, f"{args.slice_id}of{args.slice_num}_q_adv_global_correction_stats.json")
    with open(global_stats_file, "w") as f:
        json.dump(global_correction_stats, f, indent=2)

    logger.info("[GLOBAL_CORRECTION_STATS] Saved to %s", global_stats_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--slice_id", type=int, default=0)
    parser.add_argument("--slice_num", type=int, default=1, help="Evaluation part.")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--n_trajs", type=int, default=1, help="number of workers for evaluation.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="webshop",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--samples_per_depth",
        type=int,
        default=1,
        help="Num of explored tree nodes at each depth.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--disable_perturb",
        action="store_true",
        help="Whether to disable perturbation of messages.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--force_first",
        action="store_true",
        help="Set the ICL format of the environment to be first.",
    )
    parser.add_argument(
        "--apply_sigmoid",
        action="store_true",
        help="Set the ICL format of the environment to be first.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--num_icl_examples",
        type=int,
        default=1,
        help="Number of ICL examples to generate.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of the tree.",
    )
    parser.add_argument(
        "--qnet_path",
        type=str,
        default=None,
        help="Path to the QNet model.",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default='epsilon_greedy',
        help="Sampling mode for the trajectories.",
    )

    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="q_argmax",
        choices=["q_argmax", "q_argmax_oversample", "q_adv_logit_argmax", "q_adv_logit_sample"],
        help=(
            "Final candidate selection strategy. "
            "q_argmax keeps the original QLASS raw-Q selection. "
            "q_argmax_oversample uses the q_adv canonical oversampling candidate generator, but selects by raw QLASS score only. "
            "q_adv_logit_argmax/q_adv_logit_sample enable canonical candidate generation with actor output logprobs; corrected-score selection is added in the next step."
        ),
    )
    parser.add_argument(
        "--q_adv_beta",
        type=float,
        default=1.0,
        help="Weight for Q-derived normalized advantage in corrected-score selection.",
    )
    parser.add_argument(
        "--actor_logprob_coef",
        type=float,
        default=1.0,
        help="Weight for actor action-level logprob in corrected-score selection.",
    )
    parser.add_argument(
        "--q_adv_eps",
        type=float,
        default=1e-6,
        help="Epsilon for Q-score std normalization in q-advantage computation.",
    )
    parser.add_argument(
        "--q_adv_clip",
        type=float,
        default=5.0,
        help="Absolute clipping value for normalized Q advantage.",
    )
    parser.add_argument(
        "--actor_logprob_type",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Which actor logprob statistic to use in corrected-score selection.",
    )
    parser.add_argument(
        "--q_adv_sample_temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature for q_adv_logit_sample. The action is sampled from softmax(corrected_score / temperature)."
        ),
    )
    parser.add_argument(
        "--q_adv_add_candidate_suffix",
        action="store_true",
        help=(
            "If set, append an extra action-only suffix before q_adv_logit generation. "
            "Default is False to keep the prompt identical to classic QLASS generation."
        ),
    )
    parser.add_argument(
        "--candidate_oversample_factor",
        type=int,
        default=3,
        help=(
            "Oversampling factor for q_adv_logit candidate generation. "
            "The agent requests best_of_N * candidate_oversample_factor raw samples per round."
        ),
    )
    parser.add_argument(
        "--candidate_max_rounds",
        type=int,
        default=2,
        help="Maximum number of oversampling rounds for q_adv_logit candidate generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/train/explore/sft',
        help="Output directory for the trajectories and trees.",
    )
    parser.add_argument(
        "--best_of_N",
        type=int,
        default=3,
        help="Number of best trajectories to select.")

    # --- Added: QNet scoring trimming caps (to match your SGLangAgent behavior) ---
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer/SFT model path. If not set, will use agent_config['config']['tokenizer_path'] or --model_name.")
    parser.add_argument("--qnet_max_prompt_tokens", type=int, default=3800,
                        help="Max prompt tokens for QNet scoring (message-level trimming).")
    parser.add_argument("--qnet_keep_first_n", type=int, default=3,
                        help="Keep first N messages when trimming for QNet scoring.")
    parser.add_argument("--qnet_min_tail_msgs", type=int, default=4,
                        help="Keep at least this many last messages when trimming for QNet scoring.")

    args = parser.parse_args()

    if args.best_of_N <= 0:
        raise ValueError(f"--best_of_N must be positive, got {args.best_of_N}")

    if args.candidate_oversample_factor <= 0:
        raise ValueError(
            "--candidate_oversample_factor must be positive, "
            f"got {args.candidate_oversample_factor}"
        )

    if args.candidate_max_rounds <= 0:
        raise ValueError(
            f"--candidate_max_rounds must be positive, got {args.candidate_max_rounds}"
        )

    if args.q_adv_eps <= 0:
        raise ValueError(f"--q_adv_eps must be positive, got {args.q_adv_eps}")

    if args.q_adv_clip <= 0:
        raise ValueError(f"--q_adv_clip must be positive, got {args.q_adv_clip}")

    if args.q_adv_sample_temperature <= 0:
        raise ValueError(
            "--q_adv_sample_temperature must be positive, got {args.q_adv_sample_temperature}"
        )

    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)
