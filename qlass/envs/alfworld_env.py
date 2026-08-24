import re
import os
import yaml
import logging
import textworld
from typing import Tuple

from qlass.envs import BaseEnv
from qlass.tasks import AlfWorldTask
from qlass.utils import State, prompt_with_icl, prompt_without_icl

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert


logger = logging.getLogger("agent_frame")


ALFWORLD_REACT_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_REACT_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


class AlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        # self.env = task.env
        self.env = self._load_single_task_env(self.task.game_file)
        self.state = State()
        self.current_admissible_commands = []
        self.react_history = []
        self.react_current_observation = None
    
    def _load_single_task_env(self, gamefile: str):
        alfworld_data_path = "eval_agent/data/alfworld"
        with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)
        domain_randomization = config["env"]["domain_randomization"]
        if self.task.split != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        expert_type = config["env"]["expert_type"]
        training_method = config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = config["dagger"]["training"]["max_nb_steps_per_episode"]
            expert_plan = True if self.task.split == "train" else False
            if expert_plan:
                wrappers.append(AlfredExpert(expert_type))
                request_infos.extras.append("expert_plan")
        
        env_id = textworld.gym.register_games([gamefile], request_infos,
            batch_size=1,
            asynchronous=True,
            max_episode_steps=max_nb_steps_per_episode,
            wrappers=wrappers
        )
        # Launch Gym environment.
        env = textworld.gym.make(env_id)
        return env

    def get_admissible_commands(self):
        """Return admissible ALFWorld commands for the current environment state."""
        return list(getattr(self, "current_admissible_commands", []) or [])


    def build_react_actor_messages(self, history_length: int = 50):
        history_length = max(int(history_length), 0)

        admissible = [
            action for action in self.get_admissible_commands() if action != "help"
        ]
        admissible_text = "\n ".join(
            f"'{action}'" for action in admissible
        )

        if not self.react_history:
            prompt = ALFWORLD_REACT_TEMPLATE_NO_HIS.format(
                current_observation=self.react_current_observation,
                admissible_actions=admissible_text,
            )
            return [{"role": "user", "content": prompt.strip()}]

        marker = "Your task is to:"
        initial_observation = self.task.observation

        if marker in initial_observation:
            task_description = initial_observation.split(
                marker, 1
            )[1].strip()
        else:
            task_description = initial_observation.strip()

        recent_history = self.react_history[-history_length:] if history_length > 0 else []

        first_step = len(self.react_history) - len(recent_history) + 1

        history_lines = []
        for offset, (observation, action) in enumerate(recent_history):
            step_num = first_step + offset
            history_lines.append(
                f"[Observation {step_num}: '{observation}', Action {step_num}: '{action}']"
            )

        prompt = ALFWORLD_REACT_TEMPLATE.format(
            task_description=task_description,
            step_count=len(self.react_history),
            history_length=len(recent_history),
            action_history="\n".join(history_lines),
            current_step=len(self.react_history) + 1,
            current_observation=self.react_current_observation,
            admissible_actions=admissible_text,
        )

        return [{"role": "user", "content": prompt.strip()}]



    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()

        # New ReAct/AdaMEM format.
        match = re.search(
            r"<action>\s*(.*?)\s*</action>",
            llm_output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Legacy QLASS format.
        match = re.search(r"Action:\s?(.*)", llm_output, flags=re.DOTALL)
        if match:
            return match.group(1).strip()

        raise ValueError("Could not parse action from LLM output")

    def conduct_action(self, action: str):
        observation, reward, done, info = self.env.step([action])

        self.current_admissible_commands = list(
            (info.get("admissible_commands") or [[]])[0]
        )

        observation, reward, done = process_ob(observation[0]), info["won"][0], done[0]
        return observation, reward, done

    def step(self, llm_output: str) -> Tuple[str, State]:
        # import pdb; pdb.set_trace()
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        react_observation_before = self.react_current_observation
        try:
            action = self.parse_action(llm_output)
            observation, reward, done = self.conduct_action(action)

            self.react_history.append(
                (react_observation_before, action)
            )
            self.react_current_observation = observation
        except Exception as e:
            # logger.debug(f"Agent failed with error: {e}")
            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = "Observation: Error Input. Your response must contain an action inside <action>...</action> tags."
            self.react_history.append(
                (react_observation_before, "__invalid_action__")
            )
            self.react_current_observation = observation
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state


        observation = f"Observation: {observation}"
        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if done:
            self.state.finished = True
            self.state.success = bool(reward)
            self.state.reward = reward

            if self.state.success:
                self.state.terminate_reason = "success"
            elif self.state.steps >= self.max_steps:
                self.state.terminate_reason = "max_steps"
            else:
                self.state.terminate_reason = "env_done"

        elif self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward
        return observation, self.state

    def reset(self,num_icl_examples=1) -> Tuple[str, State]:
        self.state = State()
        self.state.error = self.task.game_file
        cur_task = self.task.observation

        self.react_history = []
        self.react_current_observation = cur_task

        #observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, num_icl_examples)
        if num_icl_examples > 0:
            observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, num_icl_examples)
        else:
            observation, messages = prompt_without_icl(self.instruction, cur_task)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        try:
            _, info = self.env.reset()
            self.current_admissible_commands = list(
                (info.get("admissible_commands") or [[]])[0]
            )
        except Exception:
            self.current_admissible_commands = []

        return observation, self.state
