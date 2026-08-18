import json
import re
import os
import argparse
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import os
root_dir = os.getcwd()
from pathlib import Path
import sys
sys.path.append(root_dir)
sys.path.append((Path(root_dir).parent))
import networkx as nx
import pickle
import copy
import numpy as np
from rich import print
import qlass.tasks as tasks
import qlass.agents as agents
import qlass.envs as envs
from typing import Dict, Any
import logging
from colorama import Fore
import queue
logger = logging.getLogger("agent_frame")

class TreeNode:
    _id_counter = 0
    def __init__(self, state, action,reward=0, q_value=0, critic_state=None, critic_action=None,):
        self.id = TreeNode._get_next_id() 
        self.state = state # list of messages
        self.action = action # action of the current node
        self.reward = reward
        self.q_value = q_value
        self.children = []

        # Exact state/action representation used to train the critic.
        # None for legacy exploration.
        self.critic_state = critic_state
        self.critic_action = critic_action
        
    @classmethod
    def _get_next_id(cls):
        current_id = cls._id_counter
        cls._id_counter += 1
        return current_id
    def add_child(self, child_node):
        self.children.append(child_node)

    def update_q_values(self, gamma=0.9):
        if not self.children:
            self.q_value = self.reward if self.reward is not None else 0
        else:
            for child in self.children:
                child.update_q_values(gamma)
            self.q_value = self.reward + gamma * max(child.q_value for child in self.children)

    def update_rewards(self):
        if not self.children:
            if self.reward is None:
                self.reward = 0
            return [self.reward]
        else:
            leaf_rewards = []
            for child in self.children:
                leaf_rewards.extend(child.update_rewards())
            self.reward = sum(leaf_rewards) / len(leaf_rewards)
            return leaf_rewards
        
    def print_info(self):
        print(f"State: {self.state}")
        print(f"Action: {self.action}")
        print(f"Reward: {self.reward}")
        print(f"Q-value: {self.q_value}")

    def visualize(self, level=0):
        indent = "    " * level
        # import ipdb; ipdb.set_trace()
        #print(f"Level {level}, {indent}Node(State: {self.state}, Action: {self.action}, Reward: {self.reward}, Q-value: {self.q_value})")
        for child in self.children:
            child.visualize(level + 1)

    def visualize_graph(self, G=None, pos=None, parent_name=None, level=0, horiz_spacing=1):
        if G is None:
            G = nx.DiGraph()
            pos = {}
        
        node_name = str(id(self))
        short_label = (self.action[:5] + '...') if len(self.action) > 5 else self.action
        short_label = ""
        node_label = f"{short_label}\nR: {round(self.reward,2)}\nQ: {round(self.q_value,2)}"
        G.add_node(node_name, label=node_label)
        
        # Compute horizontal position by counting nodes at this level
        if level not in pos:
            pos[level] = 0  # Initialize counter for this level
        x = pos[level] * horiz_spacing
        pos[node_name] = (x, -level)
        pos[level] += 1  # Increment counter at this level

        if parent_name:
            G.add_edge(parent_name, node_name)
        
        for child in self.children:
            child.visualize_graph(G, pos, node_name, level + 1, horiz_spacing)

        return G, {node: coords for node, coords in pos.items() if isinstance(coords, tuple)}  # Filter to keep only coordinates
    
    def child_num(self):
        return len(self.children)


def _messages_to_fastchat(messages):
    role_map = {
        "user": "human",
        "assistant": "gpt",
    }

    conversations = []
    for message in messages:
        role = message["role"]
        if role not in role_map:
            raise ValueError(
                f"Unsupported role for Q data: {role!r}"
            )

        conversations.append(
            {
                "from": role_map[role],
                "value": str(message["content"]).strip(),
            }
        )

    return conversations


def _convert_expert_action_to_react(env, raw_action):
    raw_action = str(raw_action).strip()

    # Already in the new format.
    if re.search(
        r"<action>\s*.*?\s*</action>",
        raw_action,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        return raw_action

    action = env.parse_action(raw_action)

    thought_match = re.search(
        r"Thought:\s*(.*?)\s*Action:",
        raw_action,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if thought_match:
        thought = thought_match.group(1).strip()
        return (
            f"<think>{thought}</think>\n"
            f"<action>{action}</action>"
        )

    return f"<action>{action}</action>"


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

    print(f"Average Node Reward: {average_reward}")
    print(f"Min Reward: {min_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Average Q Value: {average_q_value}")
    print(f"Min Q Value: {min_q_value}")
    print(f"Max Q Value: {max_q_value}")


MAX_TURNS={"webshop":5,"sciworld":18,"alfworld":18}


def _build_actor_messages(env, state, args):
    if args.alfworld_react_prompt:
        return env.build_react_actor_messages(
            history_length=args.alfworld_history_length
        )

    return copy.deepcopy(state.history)


def _get_node_replay_actions(node):
    """Return the exact sequence of actions leading to this tree node."""

    actions = []

    for message in node.state:
        if (
            isinstance(message, dict)
            and message.get("from") == "gpt"
        ):
            value = str(message.get("value", "")).strip()

            # Technical assistant message from
            # prompt_without_icl().
            if value and value != "OK":
                actions.append(value)

    if isinstance(node.action, dict):
        current_action = str(node.action.get("value", "")).strip()

        if current_action:
            actions.append(current_action)

    return actions


def _load_excluded_task_ids(tree_dir):
    if not tree_dir:
        return set()

    if not os.path.isdir(tree_dir):
        raise ValueError(f"Exclude tree directory does not exist: {tree_dir}")

    excluded_ids = set()

    for filename in sorted(os.listdir(tree_dir)):
        if not filename.endswith("_tree.pkl"):
            continue

        # Do not accidentally consume the output of construct_q_data.py.
        if filename == "combined_tree.pkl":
            continue

        path = os.path.join(tree_dir,filename)

        with open(path, "rb") as f:
            trees = pickle.load(f)

        for entry in trees:
            if isinstance(entry, dict) and "id" in entry:
                excluded_ids.add(str(entry["id"]))

    return excluded_ids


def main(args):

    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")

    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max_tasks must be positive.")

    if args.alfworld_react_prompt:
        if args.exp_config != "alfworld":
            raise ValueError(
                "--alfworld_react_prompt is supported only for ALFWorld."
            )

        if args.num_icl_examples != 0:
            raise ValueError(
                "Qwen ALFWorld ReAct exploration requires --num_icl_examples=0."
            )

    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name
        agent_config['config']['batch_size'] = args.eval_batch_size

    policy_overrides = {
        "server_address": args.policy_server_address,
        "model_name": args.model_name,
        "tokenizer_path": args.tokenizer_path,
        "max_prompt_tokens": args.policy_max_prompt_tokens,
        "max_new_tokens": args.policy_max_new_tokens,
        "temperature": args.policy_temperature,
        "top_p": args.policy_top_p,
        "top_k": args.policy_top_k,
        "min_p": args.policy_min_p,
        "presence_penalty": args.policy_presence_penalty,
    }

    for key, value in policy_overrides.items():
        if value is not None:
            agent_config["config"][key] = value

    if args.alfworld_react_prompt:
        agent_config["config"]["response_format_reminder"] = ""

        extra_request_body = dict(
            agent_config["config"].get("extra_request_body", {}) or {}
        )
        extra_request_body["min_tokens"] = 128

        agent_config["config"]["extra_request_body"] = extra_request_body

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
        #env_config['env'] = ScienceWorldEnv("", serverPath=f"{root_dir}/envs/scienceworld/scienceworld.jar", envStepLimit=200)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.slice_num, args.slice_id) 
    
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    tree_file = args.output_dir + f'{args.slice_id}of{args.slice_num}_0shot_tree.pkl'
    traj_file = args.output_dir + f'{args.slice_id}of{args.slice_num}_0shot_traj.jsonl'
    
    done_task_id = []
    loaded_sft_trajs = json.load(open(f'{root_dir}/data/train/{args.exp_config}/{args.exp_config}_sft.json'))
    print(f"Loaded SFT examples from {root_dir}/data/train/{args.exp_config}/{args.exp_config}_sft.json with #trajs: {len(loaded_sft_trajs)}")
    N_SFT_EXAMPLES = len(loaded_sft_trajs)

    print(f"Total number of SFT examples: {N_SFT_EXAMPLES}")
    if args.exp_config !='alfworld':
        sft_trajs =  loaded_sft_trajs[int(N_SFT_EXAMPLES//args.slice_num*args.slice_id): int(N_SFT_EXAMPLES//args.slice_num*(args.slice_id+1))]
        print(f"Loaded {len(sft_trajs)} SFT examples for slice {args.slice_id} of {args.slice_num}")
    else:
        # We need to find the intersection of the sft trajectories and the environment trajectories
        sft_traj_trail = [traj['game_file'].split('/json_2.1.1/train/')[-1] for traj in loaded_sft_trajs]
        sft_traj_set = set(sft_traj_trail)

    excluded_task_ids = _load_excluded_task_ids(
        args.exclude_tree_dir
    )
    if excluded_task_ids:
        print(f"Loaded {len(excluded_task_ids)} task ids to exclude.")

    all_tasks, n_tasks = task_class.load_tasks(args.split, args.slice_num, args.slice_id)

    total_examples = []
    total_trees = []
    total_reward_cnts = []
    processed_tasks = 0
    with logging_redirect_tqdm():
        pbar_total = args.max_tasks if args.max_tasks is not None else n_tasks
        pbar = tqdm(total=pbar_total)
        cnt = 0
        for i, task in enumerate(all_tasks):
            if args.debug and i >= 2:
                break
            print("enter iteration")
            if args.exp_config == 'alfworld' and task.game_file.split('/json_2.1.1/train/')[-1].split('/game.tw-pddl')[0] not in sft_traj_set:
                continue

            # Find the expert trajectory for the current task
            expert_traj = None
            if args.exp_config == 'webshop':
                assert sft_trajs[cnt]['id'] == task.task_id
                id = task.session_id
                expert_traj = sft_trajs[cnt]['conversations']
            elif args.exp_config == 'alfworld':
                for traj in loaded_sft_trajs:
                    if traj['game_file'].split('/json_2.1.1/train/')[-1]== task.game_file.split('/json_2.1.1/train/')[-1].split('/game.tw-pddl')[0]:
                        expert_traj = traj['conversations']
                        break
                id = task.game_file.split('/json_2.1.1/train/')[-1].split('/game.tw-pddl')[0]
            elif args.exp_config=='sciworld':
                assert str(sft_trajs[cnt]['id'].split('_')[-1]) == str(task.task_id.split('_')[-1])
                id = sft_trajs[cnt]['id']
                expert_traj = sft_trajs[cnt]['conversations']
            else:
                raise NotImplementedError(f"Unsupported task: {args.exp_config}")
            assert expert_traj != None

            if str(id) in excluded_task_ids:
                continue

            if args.max_tasks is not None and processed_tasks >= args.max_tasks:
                break

            processed_tasks += 1

            ds = args.exp_config
            
            # Load the environment
            env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
            # env.icl_format = 'first' # first for treating the icl_example as prompt instead of several turns of conversations
            env.max_steps = (
                args.max_steps
                if args.max_steps is not None
                else MAX_TURNS[args.exp_config]
            )
            if env.icl_format == 'first':
                start_i = 1
            elif env.icl_format == 'conversation':
                start_i = 2

            if args.exp_config == 'webshop':
                expert_reward = expert_traj['reward']
            else:
                expert_reward = 0.
            
            cnt += 1
            # All explored trajectories for the current task
            all_trajs = []
            print(f"Task {i}")
            if id in done_task_id or str(id) in done_task_id:
                continue

            # Load the sft traj&environment by running the expert actions
            observation, sft_state = env.reset(num_icl_examples=args.num_icl_examples) 
            init_msg = observation
            logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")    
            cur_step = 1

            expert_action_list = [conv["value"] for conv in expert_traj if conv["from"] == "gpt"]

            # Старые ALFWorld SFT trajectories обычно начинаются с технического assistant message "OK".
            if (
                expert_action_list
                and expert_action_list[0].strip() == "OK"
            ):
                expert_actions = expert_action_list[1:]
            else:
                expert_actions = expert_action_list

            if args.alfworld_react_prompt:
                expert_actions = [_convert_expert_action_to_react(env, action) for action in expert_actions]

            expert_critic_inputs = []
            expert_executed_actions = []

            for expert_action in expert_actions:
                if sft_state.finished:
                    break

                # Save the exact PRE-ACTION ReAct state that the critic should later evaluate.
                if args.alfworld_react_prompt:
                    expert_critic_inputs.append(
                        env.build_react_actor_messages(history_length=args.alfworld_history_length)
                    )

                expert_executed_actions.append(expert_action)

                observation, sft_state = env.step(expert_action)

            # Expert input was exhausted before the environment
            # reported a terminal state.
            if not sft_state.finished:
                sft_state.success = False
                sft_state.finished = True
                sft_state.terminate_reason = "exceeding maximum input length"
                sft_state.reward = expert_reward

                logger.info(
                    "Task finished in %d steps. Success: %s. Reward: %s",
                    sft_state.steps,
                    sft_state.success,
                    sft_state.reward,
                )

            root = TreeNode(state=init_msg, action='No Action (Root)',reward=0)
            node_queue = queue.Queue()
            node_queue.put(root)
            depth_queue = queue.Queue() 
            depth_queue.put(1) # Explore root node at depth 1 
            n_success = 0
            n_traj = 0

            while not node_queue.empty():
                node = node_queue.get()
                depth = depth_queue.get()

                explore_samples = args.samples_per_depth
                
                new_action_list = [child.action for child in node.children]

                # Expand child action to samples_per_depth
                for _ in range(len(new_action_list), explore_samples):
                    # observation, state = env.reset(args.num_icl_examples)
                    # cur_step = 1
                    # # Get the corresponding state of the current node
                    # if depth > 1:
                    #     for cur_step in range(start_i, depth + start_i - 1):
                    #         if cur_step < depth-1:
                    #             action = node.state[cur_step*2-1]['value']
                    #         else:
                    #             action = node.action['value']
                    #         observation, state = env.step(action)
                    # assert not state.finished
                    observation, state = env.reset(args.num_icl_examples)
                    reached_terminal = False

                    if depth > 1:
                        replay_actions = _get_node_replay_actions(node)

                        for action in replay_actions:
                            observation, state = env.step(action)

                            if state.finished:
                                reached_terminal = True
                                break

                    if reached_terminal:
                        continue

                    base_input = _build_actor_messages(
                        env=env,
                        state=state,
                        args=args,
                    )

                    input = copy.deepcopy(base_input)
                    if len(new_action_list) > 0:
                        # Make sure the agent does not repeat the already explored action
                        if len(new_action_list) == 1:
                            explore_add_prompt = f"\nPlease provide another reasonable response different from '{new_action_list[0]}'."
                            if args.exp_config=='webshop':
                                explore_add_prompt +=  "If the previous action was a search, your new action should be searching for different content. If it was a click, your new action should be clicking on a different item."
                        else:
                            explore_add_prompt =f"\nYou have given the the following answers. \n"
                            for i in range(len(new_action_list)):
                                explore_add_prompt += f"Answer {i+1}: '{new_action_list[i]}'\n"
                            explore_add_prompt += "Please provide a reasonable response different from previous answers."
                            if args.exp_config=='webshop':
                                explore_add_prompt += "If the previous actions were 'search', your new action should be searching for different content. If previous actions were 'click', your new action should be clicking on a different item."

                        input[-1]["content"] += explore_add_prompt

                    action = agent(input)

                    new_action_list.append(action)

                    # Check if the action is the same as the previous action, if so, skip it from expansion
                    # In this case, the number child node might not be the same as the samples_per_depth
                    if len(new_action_list) > 1 and new_action_list[-1] == new_action_list[-2]:
                        continue
                    
                    _, new_state = env.step(action)
                    # cur_state = new_state.to_dict()['conversations']
                    try:
                        cur_state = new_state.to_dict()['conversations']
                    except Exception:
                        print("BAD HISTORY:")
                        for j, msg in enumerate(new_state.history):
                            print(j, msg)
                        raise

                    critic_state = None
                    critic_action = None

                    if args.alfworld_react_prompt:
                        critic_state = _messages_to_fastchat(base_input)
                        critic_action = {
                            "from": "gpt",
                            "value": action.strip(),
                        }

                    new_node = TreeNode(
                        state=cur_state[:-2],
                        action=cur_state[-2],
                        reward=new_state.reward,
                        critic_state=critic_state,
                        critic_action=critic_action,
                    )
                    assert isinstance(cur_state[-2],dict) and cur_state[-2]['from']=='gpt'
                    node.add_child(new_node)

                    if not new_state.finished :
                        new_depth = depth + 1
                        if new_depth <= args.max_depth:
                            node_queue.put(new_node)
                            depth_queue.put(new_depth)
                        # Roll out the trajectory
                        current_node = new_node
                        current_depth = new_depth + 1
                        initial_new_node = new_node
                        while not new_state.finished:
                            base_input = _build_actor_messages(
                                env=env,
                                state=new_state,
                                args=args,
                            )
                            action = agent(base_input)
                            _, new_state = env.step(action)
                            # current_depth +=1
                            new_node = TreeNode(
                                state=new_state.to_dict()['conversations'][:-2],
                                action=new_state.to_dict()['conversations'][-2],
                                reward=new_state.reward,
                                critic_state=_messages_to_fastchat(base_input) if args.alfworld_react_prompt else None,
                                critic_action={"from": "gpt", "value": action.strip()} if args.alfworld_react_prompt else None,
                            )
                            assert isinstance(new_state.to_dict()['conversations'][-2],dict) and new_state.to_dict()['conversations'][-2]['from']=='gpt'
                            current_node.add_child(new_node)
                            current_node = new_node
                            if new_state.finished:
                                all_trajs.append(
                                {
                                    'dataset': ds,
                                    'id': id,
                                    'conversations': new_state.to_dict()['conversations'],
                                    'reward': new_state.reward,
                                    'success': new_state.success,
                                })
                                n_traj += 1
                                n_success += 1 if new_state.reward == 1.0 else 0
                        added_node = initial_new_node.children[0]

                        if new_state.reward > 0.01 and current_depth <= args.max_depth and current_depth > args.min_prune_depth: # prune the branches that lead to low reward
                            while current_depth <= args.max_depth and len(added_node.children) > 0:
                                node_queue.put(added_node)
                                depth_queue.put(current_depth)
                                current_depth += 1
                                added_node = added_node.children[0]
                    else:
                        all_trajs.append(
                            {
                                'dataset': ds,
                                'id': id,
                                'conversations': new_state.to_dict()['conversations'],
                                'reward': new_state.reward,
                                'success': new_state.success,
                            }
                        )
                        n_traj += 1
                        n_success += 1 if new_state.reward == 1.0 else 0

            # 2nd try: sft_traj does get involved into expanding nodes
            current_node = root
            # adding the sft traj and a branch

            for i in range(start_i, sft_state.steps+start_i):
                # new_state = sft_state.to_dict()['conversations'][:i*2]
                # new_node = TreeNode(state=new_state[:-1], action=new_state[-1], reward=sft_state.reward)
                # assert isinstance(new_state[-1],dict) and new_state[-1]['from']=='gpt'
                conv = sft_state.to_dict()['conversations'][:i*2]

                expert_idx = i - start_i

                if len(conv) >= 2 and conv[-1]['from'] == 'human' and conv[-2]['from'] == 'gpt':
                    new_node = TreeNode(
                        state=conv[:-2],
                        action=conv[-2],
                        reward=sft_state.reward,
                        critic_state=_messages_to_fastchat(expert_critic_inputs[expert_idx]) if args.alfworld_react_prompt else None,
                        critic_action={"from": "gpt", "value": expert_executed_actions[expert_idx]} if args.alfworld_react_prompt else None,
                    )
                elif len(conv) >= 1 and conv[-1]['from'] == 'gpt':
                    new_node = TreeNode(
                        state=conv[:-1],
                        action=conv[-1],
                        reward=sft_state.reward,
                        critic_state=_messages_to_fastchat(expert_critic_inputs[expert_idx]) if args.alfworld_react_prompt else None,
                        critic_action={"from": "gpt", "value": expert_executed_actions[expert_idx]} if args.alfworld_react_prompt else None,
                    )
                else:
                    raise ValueError(f"Unexpected conversation suffix: {conv[-4:]}")

                if i > start_i and i <= args.max_depth:
                    _, bro_state = env.reset(args.num_icl_examples)

                    if args.alfworld_react_prompt:
                        prefix_actions = expert_executed_actions[:expert_idx]
                    else:
                        prefix_actions = [
                            m["value"] for m in conv if m["from"] == "gpt"
                        ][:-1]

                    for action in prefix_actions:
                        _, bro_state = env.step(action)
                        if bro_state.finished:
                            break

                    if bro_state.finished:
                        continue

                    base_input = _build_actor_messages(
                        env=env,
                        state=bro_state,
                        args=args,
                    )
                    brother_action = agent(base_input)

                    _, bro_state = env.step(brother_action)
                    brother_root_node = TreeNode(
                        state=bro_state.to_dict()['conversations'][:-2],
                        action=bro_state.to_dict()['conversations'][-2],
                        reward=bro_state.reward,
                        critic_state=_messages_to_fastchat(base_input) if args.alfworld_react_prompt else None,
                        critic_action={"from": "gpt", "value": brother_action.strip()} if args.alfworld_react_prompt else None,
                    )
                    assert isinstance(bro_state.to_dict()['conversations'][-2],dict)and bro_state.to_dict()['conversations'][-2]['from']=='gpt' 
                    brother_current_node = brother_root_node

                    while not bro_state.finished:
                        base_input = _build_actor_messages(
                            env=env,
                            state=bro_state,
                            args=args,
                        )
                        brother_action = agent(base_input)
                        _,bro_state = env.step(brother_action)
                        brother_new_node = TreeNode(
                            state=bro_state.to_dict()['conversations'][:-2],
                            action=bro_state.to_dict()['conversations'][-2],
                            reward=bro_state.reward,
                            critic_state=_messages_to_fastchat(base_input) if args.alfworld_react_prompt else None,
                            critic_action={"from": "gpt", "value": brother_action.strip()} if args.alfworld_react_prompt else None,
                        )
                        assert isinstance(bro_state.to_dict()['conversations'][-2], dict) and bro_state.to_dict()['conversations'][-2]['from']=='gpt' 
                        brother_current_node.add_child(brother_new_node)
                        brother_current_node = brother_new_node

                    all_trajs.append(
                        {
                            'dataset': ds,
                            'id': id,
                            'conversations': bro_state.to_dict()['conversations'],
                            'reward': bro_state.reward,
                            'success': bro_state.success,
                        }
                    )
                    n_traj += 1
                    n_success += 1 if bro_state.reward == 1 else 0
                    current_node.add_child(brother_root_node)
                    
                current_node.add_child(new_node)
                current_node = new_node

            n_traj += 1
            n_success += 1 if sft_state.reward == 1 else 0
            
            all_trajs.append(
                {
                    'dataset':ds,
                    'id':id,
                    'conversations': sft_state.to_dict()['conversations'],
                    'reward': sft_state.reward,
                    'success': sft_state.success,
                })

                            
            print(f"Number of Successful Trajectories: {n_success} / {n_traj} ")
            print(f"Average Reward: {sum([traj['reward'] for traj in all_trajs]) / n_traj}")
            total_reward_cnts.append(len(np.unique([traj['reward'] for traj in all_trajs])))
            collect_and_print_stats(root)
            total_examples += all_trajs
            total_trees.append({'dataset':ds, 'id': id, 'tree': root})
            
            # if id % 10 == 0:
            with open(traj_file, 'w') as f:
                json.dump(total_examples, f, indent=4)
            
            # save as pickle
            with open(tree_file, 'wb') as f:
                pickle.dump(total_trees, f)
                
            done_task_id.append(id) 
            pbar.update(1)
            
        pbar.close()

    n_traj = len(total_examples)
    n_success = sum([ 1. if traj['reward']==1.0 else 0. for traj in total_examples])
    rewards = [traj['reward'] for traj in total_examples]
    print(f"Finally, The Number of Successful Trajectories: {n_success} / {n_traj} ")
    print(f"Average Reward: {sum(rewards) / n_traj}")
    print("num of reward counts:",total_reward_cnts)
    
    if args.debug:
        traj_file = f'{root_dir}/data/train/explore/debug_{args.exp_name}.jsonl'
        tree_file = f'{root_dir}/data/train/explore/debug_{args.exp_name}.pkl'
        
    with open(traj_file, 'w') as f:
        #print("total_examples:",total_examples)
        json.dump(total_examples, f, indent=4)
            
    with open(tree_file, 'wb') as f:
        pickle.dump(total_trees, f)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--slice_id", type=int, default=0)
    parser.add_argument("--slice_num", type=int, default=1, help="Evaluation part.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
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
        "--recursive",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
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
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
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
        "--min_prune_depth",
        type=int,
        default=3,
        help="Minimum depth to prune of the tree.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/train/explore/",
        help='The path to save the generated trajectories and trees.'
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of eligible tasks to process. Useful for smoke tests.",
    )

    parser.add_argument(
        "--policy_server_address",
        type=str,
        default=None,
    )
    parser.add_argument("--tokenizer_path", type=str, default=None,
        help="Tokenizer path used by the policy actor. For SGLangChatAgent with Qwen this must point to the Qwen checkpoint.",
    )
    parser.add_argument(
        "--policy_max_prompt_tokens",
        type=int,
        default=3800,
    )
    parser.add_argument(
        "--policy_max_new_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--policy_temperature",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--policy_top_p",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--policy_top_k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--policy_min_p",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--policy_presence_penalty",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--alfworld_react_prompt",
        action="store_true",
        help=(
            "Use AdaMEM-style ALFWorld ReAct actor prompt with compact observation/action history and admissible actions."
        ),
    )

    parser.add_argument(
        "--alfworld_history_length",
        type=int,
        default=50,
        help="Number of previous ALFWorld observation/action steps in ReAct prompt.",
    )

    parser.add_argument(
        "--exclude_tree_dir",
        type=str,
        default=None,
        help=(
            "Directory containing previous *_tree.pkl files. Tasks already present there will be skipped."
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
