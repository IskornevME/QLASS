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


def _force_min_max_steps(env, min_steps: int):
    """SciWorld reset can overwrite env.max_steps (e.g. from max_steps.json). Force it back."""
    cur = getattr(env, "max_steps", None)
    if cur is None:
        env.max_steps = int(min_steps)
    else:
        env.max_steps = int(max(int(cur), int(min_steps)))


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
                # for n_turn in range(turn_cap):
                for n_turn in range(env.max_steps):
                    new_state_candidates = []
                    if n_turn==0:
                        current_node=root
                    action_list = []
                    action_value_dict = []

                    # Sampling N actions and select the best one
                    for idx in range(args.best_of_N):
                        # import ipdb; ipdb.set_trace()
                        # observation, state = env.reset(num_icl_examples=args.num_icl_examples) 
                        # if n_turn >0:
                        #     for i in range(start_i,n_turn+start_i):
                        #         action = cur_traj_state.history[i*2-1]['content']
                        #         observation, state = env.step(action)
        
                        # assert state.history==cur_traj_state.history, f"state.history: {state.history}\ncur_traj_state.history: {cur_traj_state.history} mismatch"

                        def extract_assistant_action_msgs(history):
                            out = []
                            for msg in history:
                                if msg["role"] != "assistant":
                                    continue
                                text = msg["content"].strip()
                                # берем именно те сообщения, которые реально шли в env.step (с Thought/Action или numeric disambiguation)
                                if ("Action:" in text) or text.isdigit():
                                    out.append(text)
                            return out
                        
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

                        if (not idx) or args.disable_perturb:
                            cur_state_history = cur_traj_state.history
                        else:
                            assert not args.disable_perturb
                            if args.exp_config == 'webshop':
                                cur_state_history = perturb_messages(instruction,copy.deepcopy(cur_traj_state.history))
                            else:
                                # Make sure the agent does not repeat the already explored action
                                if idx == 1:
                                    explore_add_prompt = f"\nPlease provide another reasonable response different from '{action_list[0]}'."
                                else:
                                    explore_add_prompt =f"\nYou have given the the following answers. \n"
                                    for i in range(len(action_list)):
                                        explore_add_prompt += f"Answer {i+1}: '{action_list[i]}'\n"
                                    explore_add_prompt += "Please provide a reasonable response different from previous answers."
                                cur_state_history = copy.deepcopy(cur_traj_state.history)
                                cur_state_history[-1]['content'] += explore_add_prompt
                            
                        action = agent(cur_state_history)
                        action_list.append(action)
                        observation, new_state = env.step(action)
                        if new_state.finished:
                            new_state_candidates.append((new_state,new_state.reward))
                        else:
                            Q_value = evaluate_trajs_qnet_v2(qnet, tokenizer, new_state, batch_size=1, disable_tqdm=True, model_name=args.model_name, debug=args.debug)[0]
                            new_state_candidates.append((new_state,Q_value))
                            action_value_dict.append({"action": action, "value": Q_value})

                            if i < 2 and n_turn < 3:
                                print(f"[DBG] idx={idx} Q={Q_value:.4f} action={action.splitlines()[-1][:120]}")

                    # Select the best action from candidates based on the Q-values 
                    if args.sample_mode == 'epsilon_greedy':
                        if random.random() < EPSILON :  # Noted that this is not leakage, just a stopping signal from the model itself
                            # Explore: randomly select a trajectory
                            selected_traj = random.choice(new_state_candidates)
                        else:
                            # Exploit: select the best trajectory based on Q values
                            selected_traj = max(new_state_candidates, key=lambda x: x[1])
                    elif args.sample_mode == 'bon':
                        selected_traj = max(new_state_candidates, key=lambda x: x[1])
                    else:
                        raise NotImplementedError(f"We do not support the sample mode: {args.sample_mode}")
                    
                    action_value_list.append(action_value_dict)
                    # new_state, reward = selected_traj
                    # new_node = TreeNode(state=new_state.to_dict()['conversations'][:-2], action=new_state.to_dict()['conversations'][-2], reward= reward)
                    new_state, chosen_score = selected_traj  # chosen_score = Q_value (если не finished) или env reward (если finished)
                    new_node = TreeNode(
                        state=new_state.to_dict()['conversations'][:-2],
                        action=new_state.to_dict()['conversations'][-2],
                        reward=new_state.reward,   # всегда “настоящая” reward среды
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
                all_trajs.append(
                    {
                        'dataset': ds,
                        'id': task.task_id,
                        'conversations': cur_traj_state.to_dict()['conversations'],
                        'reward': cur_traj_state.reward,
                        'success': cur_traj_state.success,
                        'action_value_dict': action_value_list
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
    n_success = sum([ 1. if traj['reward']==1.0 else 0. for traj in total_examples])
    rewards = [traj['reward'] for traj in total_examples]
    print(f"Finally, The Number of Successful Trajectories: {n_success} / {n_traj} ")
    print(f"Finally, The Number of Successful Trajectories with multi evaluation: {mult_success_num} / {n_tasks} ")
    print(f"Finally, The Number of Successful Trajectories with first inference: {num_first_success} / {n_tasks} ")
    print(f"Average Reward: {sum(rewards) / n_traj}")
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

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)