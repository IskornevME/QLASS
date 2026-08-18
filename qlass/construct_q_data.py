# build Q data to train qnet
import argparse
import json
import random
import os
root_dir = os.getcwd()
from pathlib import Path
import sys
sys.path.append(root_dir)
sys.path.append((Path(root_dir).parent))
import pickle
from tqdm import tqdm
from qlass.explore_sft_agent import TreeNode

def combine_jsonl(input_files, output_file):
    combined_data = []
    for file in input_files:
        data = json.load(open(file, 'r'))
        print(len(data))
        combined_data.extend(data)

    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)


def combine_pkl(files, output_file):
    combined_data = []
    for file in files:
        with open(file, 'rb') as f:
            while True:
                try:
                    tree = pickle.load(f)
                    combined_data.append(tree)
                except EOFError:
                    break
            # combined_data = []
        print(len(combined_data))
    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)

def load_trees(file_path):
    trees = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                tree = pickle.load(f)
                trees.append(tree)
            except EOFError:
                break
    return trees

def post_process_tree(root_node):
    assert "Root" in root_node.action and len(root_node.children)>1
    # search_nodes= [child for child in node.children if "search[" in child.action and "]" in child.action]

    import re
    pattern = r'\[(.*?)\]'
    action_value = root_node.children[-1].action['value']
    match = re.search(pattern, action_value)
    if match:
        sft_search_content = match.group(1)
    merged_idx = []
    node_idx = 0
    for idx in range(len(root_node.children)-1):
        node = root_node.children[node_idx]
        action_value = node.action['value']
        match = re.search(pattern, action_value)
        if match:
            content = match.group(1)
            if content == sft_search_content: # merge sft node with explored node which search for the same content with sft
                for child in node.children:
                    root_node.children[-1].add_child(child)
                root_node.children.pop(node_idx) 
                print("merge sft node with explored node which search for the same content with sft")
                node_idx -=1
        node_idx += 1
    
    def del_repeated_child(node):
        node_action_list = [child.action['value'] for child in node.children]
        seen_actions = set()
        unique_children = []

        for child in node.children:
            action_value = child.action['value'].split("Action:")[-1].strip()
            if action_value not in seen_actions:
                seen_actions.add(action_value)  
                unique_children.append(child)
            node.children = unique_children
        for child in node.children:
            del_repeated_child(child)
    del_repeated_child(root_node)
    return root_node


def get_node_critic_conversation(
    node,
    require_critic_state=False,
):
    critic_state = getattr(
        node,
        "critic_state",
        None,
    )
    critic_action = getattr(
        node,
        "critic_action",
        None,
    )

    if critic_state is not None and critic_action is not None:
        if require_critic_state:
            if not isinstance(critic_state, list):
                raise TypeError(
                    f"critic_state must be a list, got {type(critic_state)}"
                )

            if len(critic_state) == 0:
                raise ValueError("critic_state must not be empty.")

            if critic_action.get("from") != "gpt":
                raise ValueError(
                    "critic_action must have from='gpt'."
                )

        return critic_state + [critic_action]

    if require_critic_state:
        raise ValueError(
            "Node does not contain critic_state/critic_action. Refusing to fall back to legacy state/action."
        )

    # Backward compatibility with old trees.
    return node.state + [node.action]


def collect_q_data_from_a_tree(node, require_critic_state=False):
    data = []
    def recurse(node):
        if node.action and node.q_value is not None :

            if not "Root" in node.action:
                #print("node.action",node.action)
                assert isinstance(node.action, dict) and node.action['from']=='gpt'
                traj = get_node_critic_conversation(
                    node,
                    require_critic_state=require_critic_state,
                )
   
            entry = {
                'conversations': traj,
                'label': node.q_value,
            }
            data.append(entry)
        
        for child in node.children:
            recurse(child)
    # start from children
    for child in node.children:
        recurse(child)    

    return data

def collect_r_data_from_a_tree(node, require_critic_state=False):
    data = []
    def recurse(node):
        if node.action and node.q_value is not None :

            if not "Root" in node.action:
                #print("node.action",node.action)
                assert isinstance(node.action, dict) and node.action['from']=='gpt'
                traj = get_node_critic_conversation(
                    node,
                    require_critic_state=require_critic_state,
                )
            entry = {
                'conversations': traj,
                'label': node.reward,
            }
            data.append(entry)
        
        for child in node.children:
            recurse(child)
    # start from children
    for child in node.children:
        recurse(child)    

    return data

def normalize_data(data, feature_key):
    values = [entry[feature_key] for entry in data if feature_key in entry]
    min_val, max_val = min(values), max(values)
    print("min_val",min_val)
    print("max_val",max_val)
    for entry in data:
        if feature_key in entry:
            entry[feature_key] = (entry[feature_key] - min_val) / (max_val - min_val) if max_val > min_val else 0
    return data

        
def update_depth_q_values(node,height,gamma=0.9):
    if not node.children:
        reward = node.reward if node.reward is not None else 0
        node.q_value = reward
        if height>1:
            for i in range(1,height):
                node.q_value += gamma**i*reward
    else:
        for child in node.children:
            update_depth_q_values(child,height-1,gamma=gamma)
        node.q_value = node.reward + gamma * max([child.q_value for child in node.children])
    
def collect_q_data_from_trees_unpruned(tree_pth, save_q_pth, require_critic_state=False,):
    # Collect q data from tree files
    trees = load_trees(tree_pth)
    q_data_ls = []
    for tree in tqdm(trees[0]):
        for t in tree:
            id = t['id']
            print("id",id)
            # post_processed_t = post_process_tree(t['tree'])
            post_processed_t = t['tree']
            post_processed_t.update_rewards()
            update_depth_q_values(post_processed_t,height=5,gamma=0.9)
            q_data = collect_q_data_from_a_tree(
                post_processed_t,
                require_critic_state=require_critic_state,
            )
            q_data = normalize_data(q_data, 'label')
            q_data = [{'id': id, **entry} for entry in q_data]
            q_data_ls.extend(q_data)
    with open(save_q_pth, 'w') as f:
        json.dump(q_data_ls, f, indent=4)

def collect_vanilla_q_data_from_trees_unpruned(
    tree_pth,
    save_q_pth,
    upper_num=300,
    require_critic_state=False,
    seed=42,
):
    rng = random.Random(seed)
    # Collect q data from tree files
    trees = load_trees(tree_pth)
    q_data_ls = []
    seen_task_ids = set()
    def update_vanilla_rewards(node):
        if len(node.children) > 0:
            node.reward = 0
            for child in node.children:
                update_vanilla_rewards(child)
        else:
            node.reward = node.reward
    for tree in tqdm(trees[0]):
        for t in tree:
            id = t['id']
            if id in seen_task_ids:
                raise ValueError(f"Duplicate task id in collected trees: {id!r}")

            seen_task_ids.add(id)
            # post_processed_t = post_process_tree(t['tree'])
            # post_processed_t.update_rewards()
            post_processed_t = t['tree']
            update_vanilla_rewards(post_processed_t)
            # import ipdb; ipdb.set_trace()
            post_processed_t.update_q_values(gamma=0.9)
            q_data = collect_q_data_from_a_tree(
                post_processed_t,
                require_critic_state=require_critic_state,
            )
            if upper_num is not None and len(q_data) > upper_num:
                q_data = rng.sample(q_data, upper_num)
                
            q_data = normalize_data(q_data, 'label')
            q_data = [{'id': id, **entry} for entry in q_data]
            q_data_ls.extend(q_data)

    print(
        f"[Q_DATA] tasks={len(seen_task_ids)} "
        f"examples={len(q_data_ls)} "
        f"avg_examples_per_task="
        f"{len(q_data_ls) / max(len(seen_task_ids), 1):.2f}"
    )

    with open(save_q_pth, 'w') as f:
        json.dump(q_data_ls, f, indent=4)

def collect_r_data_from_trees_unpruned(tree_pth, save_r_pth, require_critic_state=False):
    # Collect q data from tree files
    trees = load_trees(tree_pth)
    r_data_ls = []
    for tree in tqdm(trees[0]):
        for t in tree:
            id = t['id']
            # post_processed_t = post_process_tree(t['tree'])
            post_processed_t = t['tree']
            post_processed_t.update_rewards()
            r_data = collect_r_data_from_a_tree(
                post_processed_t,
                require_critic_state=require_critic_state,
            )
            
            r_data = normalize_data(r_data, 'label')
            r_data = [{'id': id, **entry} for entry in r_data]
            r_data_ls.extend(r_data)
    with open(save_r_pth, 'w') as f:
        json.dump(r_data_ls, f, indent=4)

def main(args):
    
    jsonl_files = sorted(
        os.path.join(args.data_path, data_name)
        for data_name in os.listdir(args.data_path)
        if data_name.endswith("_traj.jsonl") and data_name != "combined_traj.jsonl"
    )
    pkl_files = sorted(
        os.path.join(args.data_path, data_name)
        for data_name in os.listdir(args.data_path)
        if data_name.endswith("_tree.pkl") and data_name != "combined_tree.pkl"
    )
    if not pkl_files:
        raise FileNotFoundError(
            f"No worker *_tree.pkl files found in {args.data_path!r}"
        )
    combined_traj_file = os.path.join(args.data_path, 'combined_traj.jsonl')
    combined_tree_file = os.path.join(args.data_path, 'combined_tree.pkl')
    # Combined slices
    combine_jsonl(jsonl_files, combined_traj_file)
    combine_pkl(pkl_files, combined_tree_file)
    if args.output_path is not None:
        q_file = args.output_path
    else:
        # Legacy behavior.
        q_file = "data/train/" + args.task + "/explore_v2/" + f"{args.q_type}.jsonl"

    q_parent = os.path.dirname(q_file)
    if q_parent:
        os.makedirs(q_parent, exist_ok=True,)
    
    if args.q_type == 'vanilla':
        collect_vanilla_q_data_from_trees_unpruned(
            combined_tree_file,
            q_file,
            upper_num=args.upper_num,
            require_critic_state=args.require_critic_state,
            seed=args.seed,
        )
    elif args.q_type == 'pseudo_depth':
        collect_q_data_from_trees_unpruned(combined_tree_file, q_file, require_critic_state=args.require_critic_state,) ## pseudo children nodes
    elif args.q_type == 'reward':
        collect_r_data_from_trees_unpruned(combined_tree_file, q_file, require_critic_state=args.require_critic_state)
    else:
        raise ValueError(f"q_type {args.q_type} is not supported")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name or path')
    parser.add_argument('--task', type=str, default='webshop', help='task name')
    parser.add_argument('--data_path', type=str, default='data/train/webshop/self_explore_iter2/',help='data path')
    parser.add_argument('--q_type', type=str, default='vanilla', help='q_type')
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the resulting Q dataset. If omitted, use the legacy location.",
    )
    parser.add_argument(
        "--require_critic_state",
        action="store_true",
        help="Require critic_state/critic_action for every node instead of falling back to legacy state/action.",
    )
    parser.add_argument(
        "--upper_num",
        type=int,
        default=300,
        help="Maximum number of Q examples sampled from each task tree.",
    )
    parser.add_argument("--seed", type=int, default=42,)

    args = parser.parse_args()
    main(args)