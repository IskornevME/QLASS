import json
from abc import ABC, abstractmethod
from typing import Tuple

from eval_agent.utils.datatypes import State


class BaseEnv(ABC):
    def __init__(
        self,
        instruction_path: str,
        icl_path: str,
        icl_format: str = "first",
        max_steps: int = 10,
        **kwargs,
    ):
        with open(instruction_path) as f:
            self.instruction = f.read()
        self.raw_icl = json.load(open(icl_path))
        self.icl_format = icl_format
        self.max_steps = max_steps

    def get_task_text(self) -> str:
        return ""

    def get_current_observation(self) -> str:
        return ""

    def get_inventory(self) -> str:
        return ""

    def get_admissible_commands(self) -> list[str]:
        return []

    @abstractmethod
    def step(self, llm_output: str) -> Tuple[str, State]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[str, State]:
        pass
