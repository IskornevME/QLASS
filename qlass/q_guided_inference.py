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
import re
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
from typing import List, Dict, Any, Mapping, Optional, Sequence
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


def _force_min_max_steps(env, min_steps: int):
    """SciWorld reset can overwrite env.max_steps (e.g. from max_steps.json). Force it back."""
    cur = getattr(env, "max_steps", None)
    if cur is None:
        env.max_steps = int(min_steps)
    else:
        env.max_steps = int(max(int(cur), int(min_steps)))



def _append_jsonl_record(path: Optional[str], record: Mapping[str, Any]) -> None:
    """Append one structured diagnostics record to a JSONL log."""
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert optional numeric values to float.

    ALFWorld State.reward can be None for non-terminal states.
    """
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if np.isfinite(result) else default


def _extract_assistant_action_msgs(history: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return assistant responses that were actually submitted to the environment."""
    actions: List[str] = []
    for message in history:
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content", "")).strip()
        if ("Action:" in text) or text.isdigit():
            actions.append(text)
    return actions


def _extract_observation_for_memory(
    history: Sequence[Mapping[str, Any]],
    *,
    task_text: str,
    step_index: int,
) -> str:
    """Return the observation of the state before the selected action.

    On step 0 the state history contains the full ICL prompt. Using the task
    observation rather than the full prompt avoids matching on shared
    demonstration text instead of the current task.
    """
    if step_index == 0:
        return str(task_text).strip()

    for message in reversed(history):
        if message.get("role") != "user":
            continue
        text = str(message.get("content", "")).strip()
        return re.sub(r"^Observation:\s*", "", text, flags=re.IGNORECASE).strip()

    logger.warning(
        "[MEMORY_OBSERVATION_MISSING] step=%d; falling back to task text.",
        step_index,
    )
    return str(task_text).strip()


def _replace_action_line(raw_action: str, action_command: str) -> str:
    """Replace the Action line in a retrieved ReAct output with the current command.

    This is mainly a safety guard: in exact-match mode the command should
    already be the same, but replacing the action line prevents accidental
    execution of a stale object id if the stored raw action is imperfect.
    """
    text = str(raw_action or "").strip()
    if not text:
        return (
            "Thought: I should take an action that has worked in a similar state.\n"
            f"Action: {action_command}"
        )

    if re.search(r"Action:\s*.*", text, flags=re.IGNORECASE | re.DOTALL):
        return re.sub(
            r"Action:\s*.*",
            f"Action: {action_command}",
            text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    return f"{text}\nAction: {action_command}".strip()


def _build_memory_augmented_raw_action(
    action_command: str,
    *,
    action_format: str,
    retrieved_raw_action_exact: str = "",
) -> str:
    """Build assistant message for a memory-augmented ALFWorld command.

    Modes:
    - action_only:
        no reasoning, only the current admissible command.
    - generic_thought:
        a short generic ReAct-style thought plus the current command.
    - retrieved_thought_exact_only:
        use retrieved reasoning only when there was an exact action match in memory; otherwise fall back to generic_thought.
    """
    action_command = str(action_command).strip()

    if action_format == "action_only":
        return f"Action: {action_command}"

    if action_format == "generic_thought":
        return (
            "Thought: I should take an action that has worked in a similar state.\n"
            f"Action: {action_command}"
        )

    if action_format == "retrieved_thought_exact_only":
        if retrieved_raw_action_exact:
            return _replace_action_line(
                raw_action=retrieved_raw_action_exact,
                action_command=action_command,
            )
        return (
            "Thought: I should take an action that has worked in a similar state.\n"
            f"Action: {action_command}"
        )

    raise ValueError(f"Unsupported memory_augmented_action_format: {action_format}")

def _parse_action_for_memory(
    env: Any,
    raw_action: str,
    *,
    task_id: Any,
    attempt_id: int,
    step_id: int,
    candidate_id: int,
) -> str:
    """Parse the environment command used as an action key in memory.

    ``AlfWorldEnv.step`` already handles malformed responses. Therefore a
    parsing failure must be logged, but must not abort the whole evaluation.
    """
    try:
        return str(env.parse_action(raw_action)).strip()
    except Exception as exc:
        fallback = re.sub(r"\s+", " ", str(raw_action).strip().lower())
        logger.warning(
            "[ACTION_PARSE_FAILED] task=%r attempt=%d step=%d candidate=%d "
            "error=%s fallback=%r",
            task_id,
            attempt_id,
            step_id,
            candidate_id,
            exc,
            fallback,
        )
        return fallback

def main(args):
    total_examples = []
    total_trees = []
    successful_trajs = []
    token_count = 0
    
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


    # JitRL-style episodic memory for correction-only inference. It is kept
    # outside the task loop so global memory can accumulate across attempts
    # and tasks.
    correction_memory = None
    memory_log_file: Optional[str] = None
    if args.enable_memory_correction:
        if args.exp_config != "alfworld":
            raise ValueError(
                "Memory correction currently supports only --exp_config alfworld."
            )
        if args.memory_reward_mode != "terminal_only":
            raise ValueError(
                "The initial correction variant supports only terminal_only rewards."
            )
        if args.memory_use_env_reward_fallback:
            raise ValueError(
                "Environment reward fallback is disabled in the initial "
                "terminal-only correction variant."
            )
        if args.memory_weight < 0.0:
            raise ValueError("--memory_weight must be non-negative.")

        from qlass.alfworld_correction_memory import AlfWorldCorrectionMemory

        memory_dir = args.memory_dir or os.path.join(
            args.output_dir, "alfworld_correction_memory"
        )
        correction_memory = AlfWorldCorrectionMemory(
            base_dir=memory_dir,
            gamma=args.memory_gamma,
            top_k=args.memory_top_k,
            similarity_threshold=args.memory_threshold,
            terminal_step_penalty=args.memory_terminal_step_penalty,
            dynamic_threshold=not args.memory_disable_dynamic_threshold,
            persist=True,
        )
        if args.reset_memory:
            correction_memory.clear(delete_files=True)

        memory_log_file = args.memory_log_file or os.path.join(
            args.output_dir,
            f"{args.slice_id}of{args.slice_num}_slices_{args.sample_mode}_"
            "memory_correction_decisions.jsonl",
        )
        if args.reset_memory and os.path.exists(memory_log_file):
            os.remove(memory_log_file)
        _append_jsonl_record(
            memory_log_file,
            {
                "record_type": "run_config",
                "exp_config": args.exp_config,
                "split": args.split,
                "slice_id": args.slice_id,
                "slice_num": args.slice_num,
                "best_of_N": args.best_of_N,
                "n_trajs": args.n_trajs,
                "sample_mode": args.sample_mode,
                "qnet_path": args.qnet_path,
                "agent_model_name": args.model_name,
                "memory_dir": memory_dir,
                "memory_weight": args.memory_weight,
                "memory_gamma": args.memory_gamma,
                "memory_top_k": args.memory_top_k,
                "memory_threshold": args.memory_threshold,
                "memory_reward_mode": args.memory_reward_mode,
                "memory_terminal_step_penalty": args.memory_terminal_step_penalty,
                "memory_scope": args.memory_scope,
                "memory_dynamic_threshold": not args.memory_disable_dynamic_threshold,
                "prefer_terminal_success": args.prefer_terminal_success,
                "memory_augmented_action_format": args.memory_augmented_action_format,
                "memory_initial_stats": correction_memory.stats(),
            },
        )
        logger.info(
            "[MEMORY_ENABLED] dir=%s weight=%.6f gamma=%.4f top_k=%d "
            "threshold=%.4f scope=%s initial_stats=%s",
            memory_dir,
            args.memory_weight,
            args.memory_gamma,
            args.memory_top_k,
            args.memory_threshold,
            args.memory_scope,
            correction_memory.stats(),
        )
    
    turn_cap = MAX_TURNS[args.exp_config]   # for sciworld => 40

    done_task_id = []
    mult_success_num = 0
    num_first_success = 0
    memory_run_stats = {
        "decision_steps": 0,
        "steps_with_nonzero_correction": 0,
        "argmax_changed_by_correction": 0,
        "actual_selection_changed_by_memory": 0,
        "decisions_with_augmented_actions": 0,
        "augmented_actions_added": 0,
        "selected_augmented_actions": 0,
        "selection_changed_by_augmentation": 0,
    }
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

            if correction_memory is not None and args.memory_scope == "per_task":
                correction_memory.clear(delete_files=True)
                logger.info("[MEMORY_RESET_PER_TASK] task_id=%r", task.task_id)
            
            env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)

            print(
                f"[DBG_MAX_STEPS] BEFORE override: "
                f"args.exp_config={args.exp_config} "
                f"env_config.max_steps={env_config.get('max_steps', None)} "
                f"env.max_steps={getattr(env, 'max_steps', None)} "
                f"MAX_TURNS={MAX_TURNS.get(args.exp_config, None)}"
            )

            _force_min_max_steps(env, turn_cap)

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
            task_memory_stats = {
                "decision_steps": 0,
                "steps_with_nonzero_correction": 0,
                "argmax_changed_by_correction": 0,
                "actual_selection_changed_by_memory": 0,
                "decisions_with_augmented_actions": 0,
                "augmented_actions_added": 0,
                "selected_augmented_actions": 0,
                "selection_changed_by_augmentation": 0,
            }
            for traj_id in range(args.n_trajs):
                init_msg, cur_traj_state = env.reset(
                    num_icl_examples=args.num_icl_examples
                )
                _force_min_max_steps(env, turn_cap)
                print(
                    f"[DBG_MAX_STEPS] AFTER reset(traj{traj_id}) (forced): "
                    f"env.max_steps={getattr(env, 'max_steps', None)}"
                )

                # This list contains only actions actually selected for the
                # trajectory. Temporary candidate branches evaluated by QNet
                # are never inserted into episodic memory.
                executed_steps: List[Dict[str, Any]] = []
                action_value_list = []
                steps_with_nonzero_correction = 0
                argmax_changed_by_correction = 0
                decisions_changed_by_memory = 0

                decisions_with_augmented_actions = 0
                augmented_actions_added = 0
                selected_augmented_actions = 0
                selection_changed_by_augmentation_count = 0

                for n_turn in range(env.max_steps):
                    if n_turn == 0:
                        current_node = root

                    # Construct one memory query for the current pre-action
                    # state. It is shared by all candidate actions below.
                    task_text = str(getattr(task, "observation", "")).strip()
                    observation_before_action = _extract_observation_for_memory(
                        cur_traj_state.history,
                        task_text=task_text,
                        step_index=n_turn,
                    )
                    previous_action_commands = [
                        step["action_command"] for step in executed_steps
                    ]

                    action_list: List[str] = []
                    candidate_records: List[Dict[str, Any]] = []
                    pre_action_admissible_commands: List[str] = []

                    # Generate N candidates and compute their original QLASS
                    # scores exactly as in the baseline.
                    for idx in range(args.best_of_N):
                        observation, state = env.reset(
                            num_icl_examples=args.num_icl_examples
                        )
                        _force_min_max_steps(env, turn_cap)

                        if n_turn > 0:
                            prefix_actions = _extract_assistant_action_msgs(
                                cur_traj_state.history
                            )[:n_turn]
                            for act_text in prefix_actions:
                                observation, state = env.step(act_text)

                        replay_actions = _extract_assistant_action_msgs(state.history)
                        traj_actions = _extract_assistant_action_msgs(
                            cur_traj_state.history
                        )
                        assert replay_actions == traj_actions, (
                            f"[REPLAY ACTION MISMATCH] n_turn={n_turn}\n"
                            f"replay_actions={replay_actions}\n"
                            f"traj_actions={traj_actions}\n"
                        )

                        if idx == 0 and hasattr(env, "get_admissible_commands"):
                            pre_action_admissible_commands = env.get_admissible_commands()

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
                                # Preserve the original candidate-diversifying
                                # prompt used by QLASS for ALFWorld.
                                if idx == 1:
                                    explore_add_prompt = (
                                        "\nPlease provide another reasonable "
                                        f"response different from '{action_list[0]}'."
                                    )
                                else:
                                    explore_add_prompt = (
                                        "\nYou have given the the following answers. \n"
                                    )
                                    for action_idx, previous_action in enumerate(
                                        action_list
                                    ):
                                        explore_add_prompt += (
                                            f"Answer {action_idx + 1}: "
                                            f"'{previous_action}'\n"
                                        )
                                    explore_add_prompt += (
                                        "Please provide a reasonable response "
                                        "different from previous answers."
                                    )
                                cur_state_history = copy.deepcopy(
                                    cur_traj_state.history
                                )
                                cur_state_history[-1]["content"] += explore_add_prompt

                        raw_action = agent(cur_state_history)
                        action_list.append(raw_action)
                        action_command = _parse_action_for_memory(
                            env,
                            raw_action,
                            task_id=task.task_id,
                            attempt_id=traj_id,
                            step_id=n_turn,
                            candidate_id=idx,
                        )

                        observation_after_action, new_state = env.step(raw_action)

                        # Keep QLASS scoring unchanged. A terminal action has
                        # an observed environment value; any non-terminal
                        # action is evaluated by QNet.
                        if new_state.finished:
                            base_score = _safe_float(new_state.reward)
                            score_source = "environment_terminal_reward"
                        else:
                            base_score = float(
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
                            score_source = "qnet"
                            if i < 2 and n_turn < 3:
                                print(
                                    f"[DBG] idx={idx} Q={base_score:.4f} "
                                    f"action={raw_action.splitlines()[-1][:120]}"
                                )

                        candidate_records.append(
                            {
                                "candidate_id": idx,
                                "state": new_state,
                                "raw_action": raw_action,
                                "action_command": action_command,
                                "observation_after_action": observation_after_action,
                                "base_score": base_score,
                                "corrected_score": base_score,
                                "score_source": score_source,
                                "finished": bool(new_state.finished),
                                "success": bool(new_state.success),
                                "memory_has_support": False,
                                "memory_match_count": 0,
                                "memory_mean_return": 0.0,
                                "memory_raw_advantage": 0.0,
                                "memory_normalized_advantage": 0.0,
                                "memory_correction": 0.0,
                                "candidate_source": "policy",
                                "memory_augmented": False,
                                "memory_aug_mean_return": 0.0,
                                "memory_aug_match_count": 0,
                                "memory_aug_canonical_action": "",
                                "memory_aug_examples": [],
                                "memory_canonical_action": "",
                                "memory_action_examples": [],
                            }
                        )

                    augmentation_result: Optional[Dict[str, Any]] = None

                    if (
                        correction_memory is not None
                        and args.enable_memory_action_augmentation
                    ):
                        admissible_commands = pre_action_admissible_commands
                        if not admissible_commands:
                            logger.warning(
                                "[MEMORY_AUG_NO_ADMISSIBLE_COMMANDS] task=%r attempt=%d step=%d",
                                task.task_id,
                                traj_id,
                                n_turn,
                            )

                        augmentation_result = correction_memory.get_positive_actions_for_augmentation(
                            task_text=task_text,
                            observation=observation_before_action,
                            previous_actions=previous_action_commands,
                            admissible_commands=admissible_commands,
                            existing_candidate_actions=[
                                record["action_command"] for record in candidate_records
                            ],
                            inventory="",
                            min_mean_return=args.memory_augmentation_min_mean_return,
                            max_actions=args.memory_max_augmented_actions,
                            max_per_canonical=args.memory_aug_max_per_canonical,
                        )

                        for aug_idx, aug_action in enumerate(augmentation_result["augmented_actions"]):
                            action_command = aug_action["action_command"]
                            raw_action = _build_memory_augmented_raw_action(
                                action_command,
                                action_format=args.memory_augmented_action_format,
                                retrieved_raw_action_exact=aug_action.get("retrieved_raw_action_exact", ""),
                            )

                            # Replay to the current pre-action state, exactly as for policy candidates.
                            observation, state = env.reset(num_icl_examples=args.num_icl_examples)
                            _force_min_max_steps(env, turn_cap)

                            if n_turn > 0:
                                prefix_actions = _extract_assistant_action_msgs(
                                    cur_traj_state.history
                                )[:n_turn]
                                for act_text in prefix_actions:
                                    observation, state = env.step(act_text)

                            replay_actions = _extract_assistant_action_msgs(state.history)
                            traj_actions = _extract_assistant_action_msgs(cur_traj_state.history)
                            assert replay_actions == traj_actions, (
                                f"[REPLAY ACTION MISMATCH - MEMORY AUG] n_turn={n_turn}\n"
                                f"replay_actions={replay_actions}\n"
                                f"traj_actions={traj_actions}\n"
                            )

                            observation_after_action, new_state = env.step(raw_action)

                            if new_state.finished:
                                base_score = _safe_float(new_state.reward)
                                score_source = "environment_terminal_reward"
                            else:
                                base_score = float(
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
                                score_source = "qnet"

                            candidate_records.append(
                                {
                                    "candidate_id": len(candidate_records),
                                    "state": new_state,
                                    "raw_action": raw_action,
                                    "action_command": action_command,
                                    "observation_after_action": observation_after_action,
                                    "base_score": base_score,
                                    "corrected_score": base_score,
                                    "score_source": score_source,
                                    "finished": bool(new_state.finished),
                                    "success": bool(new_state.success),
                                    "memory_has_support": False,
                                    "memory_match_count": 0,
                                    "memory_mean_return": 0.0,
                                    "memory_raw_advantage": 0.0,
                                    "memory_normalized_advantage": 0.0,
                                    "memory_correction": 0.0,
                                    "candidate_source": "memory_augmentation",
                                    "memory_augmented": True,
                                    "memory_aug_mean_return": float(aug_action["mean_return"]),
                                    "memory_aug_match_count": int(aug_action["match_count"]),
                                    "memory_aug_canonical_action": aug_action["canonical_action"],
                                    "memory_aug_examples": aug_action["memory_action_examples"],
                                    "memory_augmented_action_format": args.memory_augmented_action_format,
                                    "memory_aug_retrieved_raw_action_exact": aug_action.get("retrieved_raw_action_exact", ""),
                                    "memory_aug_retrieved_raw_action_exact_count": int(
                                        aug_action.get("retrieved_raw_action_exact_count", 0)
                                    ),
                                    "memory_aug_retrieved_raw_action_exact_return": float(
                                        aug_action.get("retrieved_raw_action_exact_return", 0.0)
                                    ),
                                    "memory_aug_retrieved_raw_action_exact_similarity": float(
                                        aug_action.get("retrieved_raw_action_exact_similarity", 0.0)
                                    ),
                                    "memory_canonical_action": "",
                                    "memory_action_examples": [],
                                }
                            )

                    # Retrieval is performed once for the pre-action state.
                    # The memory class returns neutral advantages for actions
                    # without support and never introduces new actions.
                    trajectory_final_reward = _safe_float(cur_traj_state.reward)
                    memory_result: Optional[Dict[str, Any]] = None
                    if correction_memory is not None:
                        memory_result = correction_memory.score_candidates(
                            task_text=task_text,
                            observation=observation_before_action,
                            previous_actions=previous_action_commands,
                            candidate_actions=[
                                record["action_command"]
                                for record in candidate_records
                            ],
                            inventory="",
                        )
                        memory_scores = memory_result["candidate_scores"]
                        assert len(memory_scores) == len(candidate_records)

                        for record, memory_score in zip(
                            candidate_records, memory_scores
                        ):
                            normalized_advantage = float(
                                memory_score["normalized_advantage"]
                            )
                            correction = args.memory_weight * normalized_advantage
                            record.update(
                                {
                                    "memory_has_support": bool(
                                        memory_score["has_memory_support"]
                                    ),
                                    "memory_match_count": int(
                                        memory_score["match_count"]
                                    ),
                                    "memory_mean_return": float(
                                        memory_score["mean_return"]
                                    ),
                                    "memory_raw_advantage": float(
                                        memory_score["raw_advantage"]
                                    ),
                                    "memory_normalized_advantage": normalized_advantage,
                                    "memory_correction": correction,
                                    "corrected_score": record["base_score"] + correction,
                                    "memory_canonical_action": memory_score.get("canonical_action", ""),
                                    "memory_action_examples": memory_score.get("memory_action_examples", []),
                                }
                            )

                    base_best_idx = max(
                        range(len(candidate_records)),
                        key=lambda candidate_idx: candidate_records[
                            candidate_idx
                        ]["base_score"],
                    )
                    corrected_best_idx = max(
                        range(len(candidate_records)),
                        key=lambda candidate_idx: candidate_records[
                            candidate_idx
                        ]["corrected_score"],
                    )
                    ranking_changed_by_memory = base_best_idx != corrected_best_idx

                    # An observed successful completion is ground-truth
                    # information and takes precedence over memory correction.
                    successful_candidate_indices = [
                        candidate_idx
                        for candidate_idx, record in enumerate(candidate_records)
                        if record["finished"] and record["success"]
                    ]
                    selected_by_epsilon = False
                    terminal_success_override_applied = bool(
                        args.prefer_terminal_success
                        and successful_candidate_indices
                    )
                    if terminal_success_override_applied:
                        selected_idx = max(
                            successful_candidate_indices,
                            key=lambda candidate_idx: candidate_records[
                                candidate_idx
                            ]["base_score"],
                        )
                        selected_reason = "terminal_success_override"
                    elif args.sample_mode == "epsilon_greedy":
                        if random.random() < EPSILON:
                            selected_idx = random.randrange(len(candidate_records))
                            selected_by_epsilon = True
                            selected_reason = "epsilon_exploration"
                        else:
                            selected_idx = corrected_best_idx
                            selected_reason = (
                                "corrected_argmax"
                                if correction_memory is not None
                                else "qnet_argmax"
                            )
                    elif args.sample_mode == "bon":
                        selected_idx = corrected_best_idx
                        selected_reason = (
                            "corrected_argmax"
                            if correction_memory is not None
                            else "qnet_argmax"
                        )
                    else:
                        raise NotImplementedError(
                            f"We do not support the sample mode: {args.sample_mode}"
                        )

                    selection_changed_by_memory = bool(
                        correction_memory is not None
                        and not terminal_success_override_applied
                        and not selected_by_epsilon
                        and base_best_idx != selected_idx
                    )
                    has_nonzero_correction = any(
                        abs(record["memory_correction"]) > 1e-12
                        for record in candidate_records
                    )

                    if correction_memory is not None:
                        memory_run_stats["decision_steps"] += 1
                        task_memory_stats["decision_steps"] += 1

                        memory_run_stats["steps_with_nonzero_correction"] += int(
                            has_nonzero_correction
                        )
                        task_memory_stats["steps_with_nonzero_correction"] += int(
                            has_nonzero_correction
                        )

                        memory_run_stats["argmax_changed_by_correction"] += int(
                            ranking_changed_by_memory
                        )
                        task_memory_stats["argmax_changed_by_correction"] += int(
                            ranking_changed_by_memory
                        )

                        memory_run_stats["actual_selection_changed_by_memory"] += int(
                            selection_changed_by_memory
                        )
                        task_memory_stats["actual_selection_changed_by_memory"] += int(
                            selection_changed_by_memory
                        )

                    steps_with_nonzero_correction += int(has_nonzero_correction)
                    argmax_changed_by_correction += int(ranking_changed_by_memory)
                    decisions_changed_by_memory += int(selection_changed_by_memory)

                    selected_record = candidate_records[selected_idx]
                    num_augmented = sum(
                        int(record.get("memory_augmented", False))
                        for record in candidate_records
                    )
                    selected_is_augmented = bool(selected_record.get("memory_augmented", False))

                    # Best among only policy candidates, before adding memory candidates.
                    policy_candidate_indices = [
                        idx for idx, record in enumerate(candidate_records)
                        if record.get("candidate_source") == "policy"
                    ]
                    policy_base_best_idx = max(
                        policy_candidate_indices,
                        key=lambda idx: candidate_records[idx]["base_score"],
                    )

                    selection_changed_by_augmentation_step = bool(
                        args.enable_memory_action_augmentation
                        and selected_is_augmented
                        and policy_base_best_idx != selected_idx
                    )
                    if correction_memory is not None:
                        memory_run_stats["decisions_with_augmented_actions"] += int(num_augmented > 0)
                        task_memory_stats["decisions_with_augmented_actions"] += int(num_augmented > 0)

                        memory_run_stats["augmented_actions_added"] += num_augmented
                        task_memory_stats["augmented_actions_added"] += num_augmented

                        memory_run_stats["selected_augmented_actions"] += int(selected_is_augmented)
                        task_memory_stats["selected_augmented_actions"] += int(selected_is_augmented)

                        memory_run_stats["selection_changed_by_augmentation"] += int(
                            selection_changed_by_augmentation_step
                        )
                        task_memory_stats["selection_changed_by_augmentation"] += int(
                            selection_changed_by_augmentation_step
                        )

                    decisions_with_augmented_actions += int(num_augmented > 0)
                    augmented_actions_added += num_augmented
                    selected_augmented_actions += int(selected_is_augmented)
                    selection_changed_by_augmentation_count += int(
                        selection_changed_by_augmentation_step
                    )

                    for candidate_idx, record in enumerate(candidate_records):
                        record["selected"] = candidate_idx == selected_idx

                    # Preserve the old ``value`` field as the base QNet score,
                    # and save correction diagnostics alongside it.
                    action_value_dict = [
                        {
                            "action": record["raw_action"],
                            "value": record["base_score"],
                            "base_score": record["base_score"],
                            "corrected_score": record["corrected_score"],
                            "memory_correction": record["memory_correction"],
                            "memory_normalized_advantage": record[
                                "memory_normalized_advantage"
                            ],
                            "memory_match_count": record["memory_match_count"],
                            "selected": record["selected"],
                        }
                        for record in candidate_records
                        if record["score_source"] == "qnet"
                    ]
                    action_value_list.append(action_value_dict)

                    if correction_memory is not None:
                        _append_jsonl_record(
                            memory_log_file,
                            {
                                "record_type": "decision",
                                "task_index": i,
                                "task_id": task.task_id,
                                "attempt_id": traj_id,
                                "step_id": n_turn,
                                "task_text": task_text,
                                "observation_before_action": observation_before_action,
                                "previous_action_commands": previous_action_commands,
                                "memory_size_before": memory_result["memory_size"],
                                "num_retrieved_neighbors": len(
                                    memory_result["neighbors"]
                                ),
                                "num_above_threshold": memory_result[
                                    "num_above_threshold"
                                ],
                                "effective_similarity_threshold": memory_result[
                                    "effective_similarity_threshold"
                                ],
                                "baseline_return": memory_result["baseline_return"],
                                "best_matches": memory_result["best_matches"],
                                "base_best_candidate_id": base_best_idx,
                                "corrected_best_candidate_id": corrected_best_idx,
                                "selected_candidate_id": selected_idx,
                                "selected_reason": selected_reason,
                                "terminal_success_candidate_present": bool(
                                    successful_candidate_indices
                                ),
                                "terminal_success_override_applied": (
                                    terminal_success_override_applied
                                ),
                                "ranking_changed_by_memory": ranking_changed_by_memory,
                                "selection_changed_by_memory": selection_changed_by_memory,
                                "candidates": [
                                    {
                                        key: value
                                        for key, value in record.items()
                                        if key not in {"state", "observation_after_action"}
                                    }
                                    for record in candidate_records
                                ],
                                "augmentation_enabled": args.enable_memory_action_augmentation,
                                "num_augmented_candidates": num_augmented,
                                "selected_is_augmented": selected_is_augmented,
                                "selection_changed_by_augmentation": selection_changed_by_augmentation_step,
                                "augmentation_result": None if augmentation_result is None else {
                                    "num_positive_patterns": augmentation_result["num_positive_patterns"],
                                    "augmented_actions": augmentation_result["augmented_actions"],
                                    "positive_patterns": augmentation_result["positive_patterns"],
                                },
                            },
                        )
                        if selection_changed_by_memory:
                            logger.info(
                                "[MEMORY_CHANGED_DECISION] task=%r attempt=%d "
                                "step=%d base=%r corrected=%r",
                                task.task_id,
                                traj_id,
                                n_turn,
                                candidate_records[base_best_idx]["action_command"],
                                selected_record["action_command"],
                            )

                    new_state = selected_record["state"]
                    # ``q_value`` retains its former meaning: raw QNet score
                    # (or actual terminal reward). Hybrid values are stored
                    # in additional node attributes.
                    new_node = TreeNode(
                        state=new_state.to_dict()["conversations"][:-2],
                        action=new_state.to_dict()["conversations"][-2],
                        reward=new_state.reward,
                    )
                    new_node.q_value = float(selected_record["base_score"])
                    new_node.corrected_score = float(
                        selected_record["corrected_score"]
                    )
                    new_node.memory_correction = float(
                        selected_record["memory_correction"]
                    )
                    new_node.memory_normalized_advantage = float(
                        selected_record["memory_normalized_advantage"]
                    )
                    assert (
                        isinstance(new_state.to_dict()["conversations"][-2], dict)
                        and new_state.to_dict()["conversations"][-2]["from"] == "gpt"
                    )
                    current_node.add_child(new_node)
                    current_node = new_node

                    # This is the only point where an action becomes executed
                    # experience eligible for memory storage.
                    executed_steps.append(
                        {
                            "task_text": task_text,
                            "observation_before_action": observation_before_action,
                            "inventory_before_action": "",
                            "action_command": selected_record["action_command"],
                            "raw_action": selected_record["raw_action"],
                            "env_reward": _safe_float(new_state.reward),
                            "env_reward_missing": new_state.reward is None,
                            "base_q_score": selected_record["base_score"],
                            "memory_normalized_advantage": selected_record[
                                "memory_normalized_advantage"
                            ],
                            "corrected_score": selected_record["corrected_score"],
                        }
                    )

                    cur_traj_state = new_state
                    if cur_traj_state.finished:
                        print(
                            f"[DBG_DONE] finished at n_turn={n_turn + 1} "
                            f"env.max_steps={getattr(env, 'max_steps', None)} "
                            f"state.steps={getattr(cur_traj_state, 'steps', None)} "
                            f"terminate_reason={getattr(cur_traj_state, 'terminate_reason', None)}"
                        )
                        break

                memory_update: Optional[Dict[str, Any]] = None
                if correction_memory is not None:
                    memory_update = correction_memory.add_episode(
                        executed_steps=executed_steps,
                        success=bool(cur_traj_state.success),
                        final_reward=trajectory_final_reward,
                        task_id=task.task_id,
                        attempt_id=traj_id,
                        episode_metadata={
                            "task_index": i,
                            "steps_with_nonzero_correction": steps_with_nonzero_correction,
                            "argmax_changed_by_correction": argmax_changed_by_correction,
                            "decisions_changed_by_memory": decisions_changed_by_memory,
                            "decisions_with_augmented_actions": decisions_with_augmented_actions,
                            "augmented_actions_added": augmented_actions_added,
                            "selected_augmented_actions": selected_augmented_actions,
                            "selection_changed_by_augmentation": selection_changed_by_augmentation_count,
                        },
                    )
                    _append_jsonl_record(
                        memory_log_file,
                        {
                            "record_type": "episode",
                            "task_index": i,
                            "task_id": task.task_id,
                            "attempt_id": traj_id,
                            "trajectory_length": len(executed_steps),
                            "success": bool(cur_traj_state.success),
                            "final_reward": trajectory_final_reward,
                            "steps_with_nonzero_correction": steps_with_nonzero_correction,
                            "argmax_changed_by_correction": argmax_changed_by_correction,
                            "decisions_changed_by_memory": decisions_changed_by_memory,
                            "memory_update": memory_update,
                            "memory_stats_after": correction_memory.stats(),
                            "decisions_with_augmented_actions": decisions_with_augmented_actions,
                            "augmented_actions_added": augmented_actions_added,
                            "selected_augmented_actions": selected_augmented_actions,
                            "selection_changed_by_augmentation": selection_changed_by_augmentation_count,
                        },
                    )

                trajectory_record = {
                    "dataset": ds,
                    "id": task.task_id,
                    "conversations": cur_traj_state.to_dict()["conversations"],
                    "reward": cur_traj_state.reward,
                    "success": cur_traj_state.success,
                    "action_value_dict": action_value_list,
                }
                if correction_memory is not None:
                    trajectory_record["memory_correction_summary"] = {
                        "steps_with_nonzero_correction": steps_with_nonzero_correction,
                        "argmax_changed_by_correction": argmax_changed_by_correction,
                        "decisions_changed_by_memory": decisions_changed_by_memory,
                        "memory_update": memory_update,
                        "decisions_with_augmented_actions": decisions_with_augmented_actions,
                        "augmented_actions_added": augmented_actions_added,
                        "selected_augmented_actions": selected_augmented_actions,
                        "selection_changed_by_augmentation": selection_changed_by_augmentation_count,
                    }
                all_trajs.append(trajectory_record)

                if cur_traj_state.success:
                    num_success += 1
                    if traj_id == 0:
                        num_first_success += 1


            if num_success > 1:
                mult_success_num += 1

            if correction_memory is not None:
                decision_steps = task_memory_stats["decision_steps"]
                argmax_change_rate = (
                    task_memory_stats["argmax_changed_by_correction"] / decision_steps
                    if decision_steps > 0
                    else 0.0
                )
                actual_change_rate = (
                    task_memory_stats["actual_selection_changed_by_memory"] / decision_steps
                    if decision_steps > 0
                    else 0.0
                )

                _append_jsonl_record(
                    memory_log_file,
                    {
                        "record_type": "task_summary",
                        "task_index": i,
                        "task_id": task.task_id,
                        "num_attempts": args.n_trajs,
                        "num_success": num_success,
                        "memory_stats": task_memory_stats,
                        "argmax_change_rate": argmax_change_rate,
                        "actual_selection_change_rate": actual_change_rate,
                    },
                )

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
    rewards = [_safe_float(traj.get("reward", 0.0)) for traj in total_examples]
    n_success = sum(1.0 for reward in rewards if reward == 1.0)
    average_reward = sum(rewards) / n_traj if n_traj > 0 else 0.0

    print(f"Finally, The Number of Successful Trajectories: {n_success} / {n_traj} ")
    print(f"Finally, The Number of Successful Trajectories with multi evaluation: {mult_success_num} / {n_tasks} ")
    print(f"Finally, The Number of Successful Trajectories with first inference: {num_first_success} / {n_tasks} ")
    print(f"Average Reward: {average_reward}")
    if correction_memory is not None:
        decision_steps = memory_run_stats["decision_steps"]
        argmax_change_rate = (
            memory_run_stats["argmax_changed_by_correction"] / decision_steps
            if decision_steps > 0
            else 0.0
        )
        actual_change_rate = (
            memory_run_stats["actual_selection_changed_by_memory"] / decision_steps
            if decision_steps > 0
            else 0.0
        )

        print(
            "[MEMORY_SUMMARY] "
            f"argmax_changed_by_correction="
            f"{memory_run_stats['argmax_changed_by_correction']} / {decision_steps} "
            f"({argmax_change_rate:.4f}); "
            f"actual_selection_changed_by_memory="
            f"{memory_run_stats['actual_selection_changed_by_memory']} / {decision_steps} "
            f"({actual_change_rate:.4f}); "
            f"steps_with_nonzero_correction="
            f"{memory_run_stats['steps_with_nonzero_correction']} / {decision_steps}"
        )

        _append_jsonl_record(
            memory_log_file,
            {
                "record_type": "run_summary",
                "num_trajectories": n_traj,
                "num_success": n_success,
                "average_reward": average_reward,
                "memory_stats": memory_run_stats,
                "argmax_change_rate": argmax_change_rate,
                "actual_selection_change_rate": actual_change_rate,
                "final_memory_stats": correction_memory.stats(),
            },
        )
    if args.debug:
        traj_file = f'{root_dir}/data/train/explore/debug2.jsonl'
        tree_file = f'{root_dir}/data/train/explore/debug2.pkl'
        
    with open(traj_file, 'w') as f:
        json.dump(total_examples, f)
            
    with open(tree_file, 'wb') as f:
        pickle.dump(total_trees, f)
        
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

    # --- Added: JitRL-style ALFWorld memory correction of existing QLASS candidates ---
    parser.add_argument(
        "--enable_memory_correction",
        action="store_true",
        help=(
            "Enable episodic-memory correction of QNet scores. "
            "Currently supported only for --exp_config alfworld."
        ),
    )
    parser.add_argument(
        "--memory_dir",
        type=str,
        default=None,
        help=(
            "Persistent correction-memory directory. Use a fresh directory "
            "for every experimental configuration."
        ),
    )
    parser.add_argument(
        "--memory_log_file",
        type=str,
        default=None,
        help="JSONL file for step-level memory correction diagnostics.",
    )
    parser.add_argument(
        "--reset_memory",
        action="store_true",
        help="Delete records already present in --memory_dir before this run.",
    )
    parser.add_argument(
        "--memory_weight",
        type=float,
        default=0.0,
        help="Lambda in corrected_score = qnet_score + lambda * normalized_advantage.",
    )
    parser.add_argument(
        "--memory_gamma",
        type=float,
        default=0.97,
        help="Discount factor used for terminal-only memory returns.",
    )
    parser.add_argument(
        "--memory_top_k",
        type=int,
        default=10,
        help="Maximum number of retrieved memory transitions per state.",
    )
    parser.add_argument(
        "--memory_threshold",
        type=float,
        default=0.75,
        help="Initial weighted-Jaccard threshold for episodic retrieval.",
    )
    parser.add_argument(
        "--memory_terminal_step_penalty",
        type=float,
        default=0.0,
        help=(
            "Optional linear penalty for earlier steps of a terminal-only "
            "trajectory."
        ),
    )
    parser.add_argument(
        "--memory_reward_mode",
        type=str,
        choices=["terminal_only"],
        default="terminal_only",
        help="Only terminal_only is implemented in the initial correction variant.",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--memory_use_env_reward_fallback",
        dest="memory_use_env_reward_fallback",
        action="store_true",
        help=(
            "Reserved for later experiments; not supported in the initial "
            "terminal-only correction implementation."
        ),
    )
    fallback_group.add_argument(
        "--no-memory_use_env_reward_fallback",
        dest="memory_use_env_reward_fallback",
        action="store_false",
        help="Explicitly disable environment reward fallback.",
    )
    parser.set_defaults(memory_use_env_reward_fallback=False)
    parser.add_argument(
        "--memory_scope",
        type=str,
        choices=["global", "per_task"],
        default="global",
        help=(
            "global: memory persists across tasks and attempts; "
            "per_task: memory is cleared for every new task."
        ),
    )
    parser.add_argument(
        "--memory_disable_dynamic_threshold",
        action="store_true",
        help="Use a fixed similarity threshold rather than JitRL-style decay.",
    )
    parser.add_argument(
        "--prefer_terminal_success",
        action="store_true",
        help=(
            "Optional safety ablation: if a candidate demonstrably succeeds "
            "in the environment, select it regardless of corrected score. "
            "Disabled by default to preserve the original QLASS selection "
            "behavior when memory_weight=0."
        ),
    )
    parser.add_argument(
        "--enable_memory_action_augmentation",
        action="store_true",
        help=(
            "Add promising admissible actions from episodic memory to the QLASS "
            "candidate set before QNet scoring."
        ),
    )
    parser.add_argument(
        "--memory_max_augmented_actions",
        type=int,
        default=2,
        help="Maximum number of memory-augmented actions added per decision step.",
    )
    parser.add_argument(
        "--memory_aug_max_per_canonical",
        type=int,
        default=1,
        help="Maximum number of admissible commands added per canonical memory action.",
    )
    parser.add_argument(
        "--memory_augmentation_min_mean_return",
        type=float,
        default=1e-12,
        help="Minimum mean memory return required to add an action pattern.",
    )
    parser.add_argument(
        "--memory_augmented_action_format",
        type=str,
        choices=["action_only", "generic_thought", "retrieved_thought_exact_only"],
        default="action_only",
        help=(
            "How to format memory-augmented candidates before QNet scoring. "
            "retrieved_thought_exact_only uses stored reasoning only when the "
            "current admissible command exactly matches a positive retrieved memory action; "
            "otherwise it falls back to generic_thought."
        ),
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)
