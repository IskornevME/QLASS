"""JitRL-style ALFWorld memory for QLASS score correction.

This module implements only the first hybrid variant: QLASS + correction.

Implemented:
- storage of completed executed trajectories;
- pure terminal-only memory rewards;
- Jaccard-only retrieval for ALFWorld;
- discounted returns computed at retrieval time;
- memory advantages for actions already proposed by QLASS.

Intentionally NOT implemented:
- adding retrieved memory actions to the candidate set (will be done later);
- optimistic exploration bonuses for unsupported actions;
- changing the memory correction weight over episodes.

Important:
Only actions that were actually selected and executed in a completed
trajectory may be passed to "add_episode". Candidate branches temporarily
evaluated by QLASS must never be stored as executed experience.
"""


import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredStep:
    """One executed state-action transition stored in episodic memory.

    All state-related fields describe the state BEFORE ``action`` was
    executed.

    ``future_rewards`` stores immediate rewards from this action to the end
    of the episode. The discounted return is intentionally computed later,
    during retrieval, just as in terminal-only JitRL.
    """

    episode_number: int
    task_id: Optional[str]
    attempt_id: Optional[int]
    step_index: int

    task_text: str
    observation: str
    inventory: str
    previous_actions: List[str]

    action: str
    raw_action: str
    env_reward: float

    episode_success: bool
    episode_final_reward: float

    trajectory_context: str
    current_env_info: str
    future_rewards: List[float]


class AlfWorldCorrectionMemory:
    """Persistent episodic memory supplying correction signals for QLASS.

    Parameters
    ----------
    base_dir:
        Directory where memory is persisted. Use a separate empty directory
        for each experimental configuration.

    gamma:
        Discount factor used when returns are computed during retrieval.
        The current ALFWorld-adapted JitRL CLI default is 0.97.

    top_k:
        Number of similar stored transitions retained after retrieval.

    similarity_threshold:
        Initial weighted-Jaccard threshold. If ``dynamic_threshold`` is
        enabled, the threshold decreases gradually as the current trajectory
        becomes longer, following the logic in the ALFWorld JitRL branch.

    terminal_step_penalty:
        Optional non-negative linear penalty on early actions of a trajectory.

    Notes
    -----
    - Stored and candidate actions must be parsed ALFWorld environment
      commands, e.g. ``"open fridge 1"``, not full ReAct outputs such as
      ``"Thought: ... Action: open fridge 1"``.
    - This class does not apply the final correction to QNet scores. It only
      returns advantages and retrieval diagnostics.
    """

    def __init__(
        self,
        base_dir: str,
        *,
        gamma: float = 0.97,
        top_k: int = 10,
        similarity_threshold: float = 0.75,
        terminal_step_penalty: float = 0.0,
        recent_actions_window: int = 4,
        state_similarity_weight: float = 0.7,
        history_similarity_weight: float = 0.3,
        dynamic_threshold: bool = True,
        threshold_decay: float = 0.1,
        threshold_decay_horizon: int = 20,
        persist: bool = True,
    ) -> None:
        """Initialize memory and load previously stored records, if any."""

        self.base_dir = base_dir
        self.gamma = self._validate_probability("gamma", gamma)
        self.top_k = self._validate_positive_int("top_k", top_k)
        self.similarity_threshold = self._validate_probability(
            "similarity_threshold", similarity_threshold
        )
        self.terminal_step_penalty = self._validate_non_negative_float(
            "terminal_step_penalty", terminal_step_penalty
        )
        self.recent_actions_window = self._validate_positive_int(
            "recent_actions_window", recent_actions_window
        )

        self.state_similarity_weight = float(state_similarity_weight)
        self.history_similarity_weight = float(history_similarity_weight)

        self.dynamic_threshold = bool(dynamic_threshold)
        self.threshold_decay = self._validate_non_negative_float(
            "threshold_decay", threshold_decay
        )
        self.threshold_decay_horizon = self._validate_positive_int(
            "threshold_decay_horizon", threshold_decay_horizon
        )

        self.persist = bool(persist)

        weight_sum = (
            self.state_similarity_weight + self.history_similarity_weight
        )
        if (
            self.state_similarity_weight < 0.0
            or self.history_similarity_weight < 0.0
            or not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError(
                "state_similarity_weight and history_similarity_weight "
                "must be non-negative and sum to 1.0."
            )

        os.makedirs(self.base_dir, exist_ok=True)

        # JSONL is deliberately used instead of pickle: it makes memory
        # inspectable during debugging and easier to compare between runs.
        self.episodes_path = os.path.join(self.base_dir, "episodes.jsonl")
        self.steps_path = os.path.join(self.base_dir, "step_metadata.jsonl")

        self._episodes: List[Dict[str, Any]] = self._load_jsonl(
            self.episodes_path
        )
        self._steps: List[StoredStep] = [
            self._stored_step_from_dict(item)
            for item in self._load_jsonl(self.steps_path)
        ]

        self.current_episode_number = max(
            [int(item.get("episode_number", 0)) for item in self._episodes]
            + [step.episode_number for step in self._steps],
            default=0,
        )

        logger.info(
            "Initialized ALFWorld correction memory: episodes=%d, steps=%d, "
            "gamma=%.4f, top_k=%d, threshold=%.4f.",
            len(self._episodes),
            len(self._steps),
            self.gamma,
            self.top_k,
            self.similarity_threshold,
        )

    # ------------------------------------------------------------------
    # General inspection and lifecycle helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of stored executed state-action transitions."""

        return len(self._steps)

    @property
    def num_episodes(self) -> int:
        """Return the number of complete trajectories stored in memory."""

        return len(self._episodes)

    def stats(self) -> Dict[str, Any]:
        """Return compact JSON-serializable statistics for logging."""

        return {
            "num_episodes": self.num_episodes,
            "num_successful_episodes": sum(
                bool(episode.get("success", False))
                for episode in self._episodes
            ),
            "num_stored_steps": len(self),
            "gamma": self.gamma,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "dynamic_threshold": self.dynamic_threshold,
            "terminal_step_penalty": self.terminal_step_penalty,
        }

    def clear(self, *, delete_files: bool = True) -> None:
        """Clear all memory.

        This method will be useful later for the ``per_task`` memory ablation,
        where memory must be reset when switching to a new ALFWorld task.
        """

        self._episodes.clear()
        self._steps.clear()
        self.current_episode_number = 0

        if delete_files:
            for path in (self.episodes_path, self.steps_path):
                if os.path.exists(path):
                    os.remove(path)

        logger.info("Cleared ALFWorld correction memory.")

    # ------------------------------------------------------------------
    # Writing completed executed trajectories to memory
    # ------------------------------------------------------------------

    def add_episode(
        self,
        executed_steps: Sequence[Mapping[str, Any]],
        *,
        success: bool,
        final_reward: float = 0.0,
        task_id: Optional[Any] = None,
        attempt_id: Optional[int] = None,
        episode_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store one completed trajectory using terminal-only rewards.

        Parameters
        ----------
        executed_steps:
            Ordered list containing only the transitions that were genuinely
            selected and executed in the final trajectory.

            Every element should contain:

            Required:
                ``action_command`` or ``action``:
                    Parsed ALFWorld environment command.

            Strongly recommended:
                ``task_text``:
                    Task instruction.
                ``observation_before_action`` or ``observation``:
                    Observation before executing the action.
                ``inventory_before_action`` or ``inventory``:
                    Inventory before executing the action, if available.
                ``env_reward``:
                    Environment reward after this action. This field is stored
                    only for diagnostics and is not used for terminal-only
                    memory returns.

        success:
            Whether the full episode completed the task successfully.

        final_reward:
            Final environment reward. Stored for diagnostics; terminal-only
            memory uses binary success reward exactly as the current JitRL
            terminal-only mode does.

        Returns
        -------
        dict
            Summary of what was stored, suitable for structured logging.
        """

        if not executed_steps:
            logger.warning(
                "Skipping empty ALFWorld episode: task_id=%r, attempt_id=%r.",
                task_id,
                attempt_id,
            )
            return {
                "episode_added": False,
                "episode_number": None,
                "steps_added": 0,
                "success": bool(success),
            }

        self.current_episode_number += 1
        episode_number = self.current_episode_number

        terminal_rewards = self._terminal_only_rewards(
            num_steps=len(executed_steps),
            success=bool(success),
        )

        stored_steps: List[StoredStep] = []
        previous_actions: List[str] = []

        for step_index, step in enumerate(executed_steps):
            action_source = self._safe_text(
                step.get("action_command", step.get("action", ""))
            )
            action = self.normalize_action(action_source)

            raw_action = self._safe_text(
                step.get("raw_action", action_source)
            )

            if not action:
                raise ValueError(
                    f"Episode step {step_index} does not contain "
                    "a non-empty action_command/action."
                )

            task_text = self._safe_text(step.get("task_text", ""))
            observation = self._safe_text(
                step.get(
                    "observation_before_action",
                    step.get("observation", ""),
                )
            )
            inventory = self._safe_text(
                step.get(
                    "inventory_before_action",
                    step.get("inventory", ""),
                )
            )

            trajectory_context, current_env_info = self.build_retrieval_texts(
                task_text=task_text,
                observation=observation,
                previous_actions=previous_actions,
                inventory=inventory,
            )

            stored_step = StoredStep(
                episode_number=episode_number,
                task_id=None if task_id is None else str(task_id),
                attempt_id=attempt_id,
                step_index=step_index,
                task_text=task_text,
                observation=observation,
                inventory=inventory,
                previous_actions=list(previous_actions),
                action=action,
                raw_action=raw_action,
                env_reward=self._safe_float(step.get("env_reward", 0.0)),
                episode_success=bool(success),
                episode_final_reward=self._safe_float(final_reward),
                trajectory_context=trajectory_context,
                current_env_info=current_env_info,
                future_rewards=list(terminal_rewards[step_index:]),
            )

            stored_steps.append(stored_step)

            # Only the already executed selected actions become history
            # features for the following transition.
            previous_actions.append(action)

        episode_record: Dict[str, Any] = {
            "episode_number": episode_number,
            "task_id": None if task_id is None else str(task_id),
            "attempt_id": attempt_id,
            "success": bool(success),
            "final_reward": self._safe_float(final_reward),
            "reward_mode": "terminal_only",
            "terminal_step_penalty": self.terminal_step_penalty,
            "num_steps": len(stored_steps),
            "metadata": dict(episode_metadata or {}),
        }

        self._episodes.append(episode_record)
        self._steps.extend(stored_steps)

        if self.persist:
            self._append_jsonl(self.episodes_path, episode_record)
            for stored_step in stored_steps:
                self._append_jsonl(self.steps_path, asdict(stored_step))

        logger.info(
            "Stored terminal-only ALFWorld episode #%d: "
            "task_id=%r, attempt_id=%r, steps=%d, success=%s, "
            "terminal_reward=%.1f, memory_steps=%d.",
            episode_number,
            task_id,
            attempt_id,
            len(stored_steps),
            bool(success),
            terminal_rewards[-1],
            len(self._steps),
        )

        return {
            "episode_added": True,
            "episode_number": episode_number,
            "steps_added": len(stored_steps),
            "success": bool(success),
            "terminal_reward": terminal_rewards[-1],
            "memory_steps_after": len(self._steps),
        }

    # ------------------------------------------------------------------
    # Retrieval and memory correction signals
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        *,
        task_text: str,
        observation: str,
        previous_actions: Sequence[str],
        candidate_actions: Sequence[str],
        inventory: str = "",
    ) -> Dict[str, Any]:
        """Compute memory advantages for existing QLASS candidates.

        This is the main method that will later be called from
        ``q_guided_inference.py``.

        The method deliberately does NOT:
        - add new actions from memory;
        - give optimistic bonuses to candidates not found in memory;
        - multiply correction by the current episode number.

        For every unsupported candidate action:

            raw_advantage = 0
            normalized_advantage = 0

        Parameters
        ----------
        candidate_actions:
            Parsed ALFWorld commands corresponding to actions already
            generated by the QLASS policy model.

        Returns
        -------
        dict
            Retrieval diagnostics and a ``candidate_scores`` list in exactly
            the same order as ``candidate_actions``.
        """

        retrieval_result = self.retrieve(
            task_text=task_text,
            observation=observation,
            previous_actions=previous_actions,
            inventory=inventory,
        )

        neighbors = retrieval_result["neighbors"]
        normalized_candidates = [
            self.normalize_action(action)
            for action in candidate_actions
        ]
        canonical_candidates = [
            self.canonicalize_action(action)
            for action in normalized_candidates
        ]

        if not neighbors:
            retrieval_result.update(
                {
                    "baseline_return": 0.0,
                    "action_reward_support": {},
                    "candidate_scores": [
                        self._unsupported_candidate_score(
                            input_action=raw_action,
                            normalized_action=normalized_action,
                            canonical_action=canonical_action,
                        )
                        for raw_action, normalized_action, canonical_action in zip(
                            candidate_actions,
                            normalized_candidates,
                            canonical_candidates,
                        )
                    ],
                }
            )
            return retrieval_result

        # Aggregate returns for ALL retrieved memory actions.
        #
        # Even though we correct only QLASS-proposed candidates, the baseline
        # should describe the complete retrieved local neighborhood, as in
        # the JitRL advantage computation.
        returns_by_action: Dict[str, List[float]] = defaultdict(list)
        exact_examples_by_canonical: Dict[str, List[str]] = defaultdict(list)

        for neighbor in neighbors:
            exact_action = neighbor["action"]
            canonical_action = self.canonicalize_action(exact_action)
            if not canonical_action:
                continue

            returns_by_action[canonical_action].append(
                float(neighbor["discounted_return"])
            )
            exact_examples_by_canonical[canonical_action].append(exact_action)

        all_returns = [
            discounted_return
            for action_returns in returns_by_action.values()
            for discounted_return in action_returns
        ]

        baseline_return = mean(all_returns) if all_returns else 0.0

        mean_return_by_action = {
            action: mean(action_returns)
            for action, action_returns in returns_by_action.items()
        }

        raw_advantages = {
            action: action_mean_return - baseline_return
            for action, action_mean_return in mean_return_by_action.items()
        }

        normalized_advantages = self._normalize_advantages_like_jitrl(
            raw_advantages
        )

        candidate_scores: List[Dict[str, Any]] = []

        for raw_action, normalized_action, canonical_action in zip(
            candidate_actions,
            normalized_candidates,
            canonical_candidates,
        ):
            if canonical_action not in returns_by_action:
                candidate_scores.append(
                    self._unsupported_candidate_score(
                        input_action=raw_action,
                        normalized_action=normalized_action,
                        canonical_action=canonical_action,
                    )
                )
                continue

            candidate_scores.append(
                {
                    "input_action": raw_action,
                    "normalized_action": normalized_action,
                    "canonical_action": canonical_action,
                    "has_memory_support": True,
                    "match_count": len(returns_by_action[canonical_action]),
                    "mean_return": mean_return_by_action[canonical_action],
                    "raw_advantage": raw_advantages[canonical_action],
                    "normalized_advantage": normalized_advantages[canonical_action],
                    "memory_action_examples": exact_examples_by_canonical[canonical_action][:3],
                }
            )

        retrieval_result.update(
            {
                "baseline_return": baseline_return,
                "action_reward_support": {
                    canonical_action: {
                        "returns": list(action_returns),
                        "match_count": len(action_returns),
                        "mean_return": mean_return_by_action[canonical_action],
                        "raw_advantage": raw_advantages[canonical_action],
                        "normalized_advantage": normalized_advantages[canonical_action],
                        "exact_action_examples": exact_examples_by_canonical[canonical_action][:3],
                    }
                    for canonical_action, action_returns in returns_by_action.items()
                },
                "action_key_mode": "canonical",
                "candidate_scores": candidate_scores,
            }
        )

        logger.debug(
            "Scored %d QLASS candidates using %d retrieved neighbors: "
            "baseline_return=%.6f, supported_candidates=%d.",
            len(candidate_actions),
            len(neighbors),
            baseline_return,
            sum(
                int(candidate["has_memory_support"])
                for candidate in candidate_scores
            ),
        )

        return retrieval_result

    def retrieve(
        self,
        *,
        task_text: str,
        observation: str,
        previous_actions: Sequence[str],
        inventory: str = "",
    ) -> Dict[str, Any]:
        """Retrieve similar stored steps using ALFWorld JitRL logic.

        Similarity uses:

            history_similarity = unigram multiset Jaccard over
                                 TASK + RECENT_ACTIONS

            state_similarity = 4-gram multiset Jaccard over
                               TASK + CURRENT_OBSERVATION + INVENTORY

            similarity = 0.3 * history_similarity
                       + 0.7 * state_similarity

        Only records above the current effective threshold are returned as
        neighbors.
        """

        normalized_previous_actions = [
            self.normalize_action(action)
            for action in previous_actions
            if self.normalize_action(action)
        ]

        query_trajectory_context, query_current_env_info = (
            self.build_retrieval_texts(
                task_text=task_text,
                observation=observation,
                previous_actions=normalized_previous_actions,
                inventory=inventory,
            )
        )

        effective_threshold = self._effective_threshold(
            num_previous_actions=len(normalized_previous_actions)
        )

        query_history_tokens = self._tokenize(query_trajectory_context)
        query_state_tokens = self._tokenize(query_current_env_info)

        matches: List[Dict[str, Any]] = []

        for stored_step in self._steps:
            history_similarity = self._multiset_jaccard(
                query_history_tokens,
                self._tokenize(stored_step.trajectory_context),
                ngram=1,
            )

            state_similarity = self._multiset_jaccard(
                query_state_tokens,
                self._tokenize(stored_step.current_env_info),
                ngram=4,
            )

            similarity = (
                self.history_similarity_weight * history_similarity
                + self.state_similarity_weight * state_similarity
            )

            discounted_return = self.compute_discounted_return(
                stored_step.future_rewards
            )

            matches.append(
                {
                    "episode_number": stored_step.episode_number,
                    "task_id": stored_step.task_id,
                    "attempt_id": stored_step.attempt_id,
                    "step_index": stored_step.step_index,
                    "action": stored_step.action,
                    "raw_action": stored_step.raw_action,
                    "discounted_return": discounted_return,
                    "similarity": similarity,
                    "history_similarity": history_similarity,
                    "state_similarity": state_similarity,
                    "episode_success": stored_step.episode_success,
                }
            )

        matches.sort(
            key=lambda item: (
                item["similarity"],
                item["discounted_return"],
            ),
            reverse=True,
        )

        above_threshold = [
            item
            for item in matches
            if item["similarity"] >= effective_threshold
        ]

        neighbors = above_threshold[: self.top_k]

        logger.debug(
            "Retrieved %d neighbors from %d memory steps: "
            "effective_threshold=%.4f, above_threshold=%d.",
            len(neighbors),
            len(self._steps),
            effective_threshold,
            len(above_threshold),
        )

        return {
            "query_trajectory_context": query_trajectory_context,
            "query_current_env_info": query_current_env_info,
            "memory_size": len(self._steps),
            "base_similarity_threshold": self.similarity_threshold,
            "effective_similarity_threshold": effective_threshold,
            "num_above_threshold": len(above_threshold),
            "neighbors": neighbors,
            # This diagnostic field is useful when no neighbor passes the
            # threshold and we need to understand whether the threshold is
            # overly restrictive.
            "best_matches": matches[: min(self.top_k, 3)],
        }

    # ------------------------------------------------------------------
    # Deterministic ALFWorld state representation
    # ------------------------------------------------------------------

    def build_retrieval_texts(
        self,
        *,
        task_text: str,
        observation: str,
        previous_actions: Sequence[str],
        inventory: str = "",
    ) -> tuple[str, str]:
        """Build the two textual representations used by retrieval.

        Following the ALFWorld JitRL branch:

        - ``trajectory_context`` contains TASK and recent executed actions;
        - ``current_env_info`` contains TASK, current observation, and
          optionally inventory;
        - admissible/valid actions are deliberately excluded because they are
          useful as constraints but noisy retrieval features.
        """

        task_text = self._safe_text(task_text)
        observation = self._safe_text(observation)
        inventory = self._safe_text(inventory)

        normalized_previous_actions = [
            self.normalize_action(action)
            for action in previous_actions
            if self.normalize_action(action)
        ]

        recent_actions = normalized_previous_actions[
            -self.recent_actions_window :
        ]

        trajectory_parts: List[str] = []

        if task_text:
            trajectory_parts.append(f"TASK: {task_text}")

        if recent_actions:
            trajectory_parts.append("RECENT_ACTIONS:")
            trajectory_parts.extend(
                f"{index}. {action}"
                for index, action in enumerate(recent_actions, start=1)
            )
        else:
            trajectory_parts.append("RECENT_ACTIONS: none")

        state_parts: List[str] = []

        if task_text:
            state_parts.append(f"TASK: {task_text}")

        state_parts.append(f"CURRENT_OBSERVATION: {observation}")

        if inventory:
            state_parts.append(f"INVENTORY: {inventory}")

        trajectory_context = "\n".join(trajectory_parts).strip()
        current_env_info = "\n".join(state_parts).strip()

        return trajectory_context, current_env_info

    # ------------------------------------------------------------------
    # Terminal-only reward and discounted return computation
    # ------------------------------------------------------------------

    @staticmethod
    def _terminal_only_rewards(
        *,
        num_steps: int,
        success: bool,
    ) -> List[float]:
        """Create sparse terminal immediate rewards for one episode.

        Successful episode:
            [0, 0, ..., 0, 1]

        Failed episode:
            [0, 0, ..., 0, 0]

        Discounting is deliberately NOT applied here.
        """

        rewards = [0.0 for _ in range(num_steps)]

        if rewards:
            rewards[-1] = 1.0 if success else 0.0

        return rewards

    def compute_discounted_return(
        self,
        future_rewards: Sequence[float],
    ) -> float:
        """Compute return from one stored action at retrieval time.

        For pure terminal-only memory without a step penalty:

            G_j = gamma ** (T - 1 - j) * R_final

        With the optional linear step penalty:

            G_j = gamma ** (T - 1 - j) * R_final
                - terminal_step_penalty * (T - 1 - j)

        The penalty is deliberately not additionally discounted.
        """

        discounted_return = sum(
            (self.gamma ** offset) * self._safe_float(reward)
            for offset, reward in enumerate(future_rewards)
        )

        if self.terminal_step_penalty > 0.0:
            num_later_actions = max(0, len(future_rewards) - 1)
            discounted_return -= (
                self.terminal_step_penalty * num_later_actions
            )

        return discounted_return

    def get_positive_actions_for_augmentation(
        self,
        *,
        task_text: str,
        observation: str,
        previous_actions: Sequence[str],
        admissible_commands: Sequence[str],
        existing_candidate_actions: Sequence[str],
        inventory: str = "",
        min_mean_return: float = 1e-12,
        max_actions: int = 2,  # сколько максимум кандидатов добавляем
        max_per_canonical: int = 1,  # сколько примеров на канон мы добавляем в кандидаты 
    ) -> Dict[str, Any]:
        """Return promising current admissible commands matched via memory.

        This method does not return raw actions from old episodes. It uses old actions only as canonical patterns and instantiates them through current
        admissible commands.
        """

        retrieval_result = self.retrieve(
            task_text=task_text,
            observation=observation,
            previous_actions=previous_actions,
            inventory=inventory,
        )

        neighbors = retrieval_result["neighbors"]

        existing_exact = {
            self.normalize_action(action)
            for action in existing_candidate_actions
        }

        admissible_by_canonical: Dict[str, List[str]] = defaultdict(list)
        for command in admissible_commands:
            exact = self.normalize_action(command)
            canonical = self.canonicalize_action(exact)
            if not exact or exact in existing_exact:
                continue
            admissible_by_canonical[canonical].append(exact)

        returns_by_canonical: Dict[str, List[float]] = defaultdict(list)
        examples_by_canonical: Dict[str, List[str]] = defaultdict(list)
        raw_actions_by_exact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for neighbor in neighbors:
            memory_action = neighbor["action"]
            canonical = self.canonicalize_action(memory_action)
            if canonical not in admissible_by_canonical:
                continue

            discounted_return = float(neighbor["discounted_return"])

            returns_by_canonical[canonical].append(discounted_return)
            examples_by_canonical[canonical].append(memory_action)

            raw_action = self._safe_text(neighbor.get("raw_action", ""))
            if raw_action and discounted_return > min_mean_return:
                raw_actions_by_exact[memory_action].append(
                    {
                        "raw_action": raw_action,
                        "discounted_return": discounted_return,
                        "similarity": float(neighbor.get("similarity", 0.0)),
                    }
                )

        positive_patterns = []
        for canonical, returns in returns_by_canonical.items():
            mean_return = mean(returns)
            if mean_return <= min_mean_return:
                continue

            positive_patterns.append(
                {
                    "canonical_action": canonical,
                    "mean_return": mean_return,
                    "match_count": len(returns),
                    "memory_action_examples": examples_by_canonical[canonical][:3],
                    "candidate_commands": admissible_by_canonical[canonical],
                }
            )

        positive_patterns.sort(
            key=lambda item: (item["mean_return"], item["match_count"]),
            reverse=True,
        )

        augmented_actions = []
        used_exact = set(existing_exact)

        for pattern in positive_patterns:
            added_for_pattern = 0

            for command in pattern["candidate_commands"]:
                exact = self.normalize_action(command)
                if exact in used_exact:
                    continue

                exact_raw_actions = sorted(
                    raw_actions_by_exact.get(exact, []),
                    key=lambda item: (
                        item["discounted_return"],
                        item["similarity"],
                    ),
                    reverse=True,
                )
                best_exact_raw_action = (
                    exact_raw_actions[0]["raw_action"]
                    if exact_raw_actions
                    else ""
                )

                augmented_actions.append(
                    {
                        "action_command": exact,
                        "canonical_action": pattern["canonical_action"],
                        "mean_return": pattern["mean_return"],
                        "match_count": pattern["match_count"],
                        "memory_action_examples": pattern["memory_action_examples"],
                        "retrieved_raw_action_exact": best_exact_raw_action,
                        "retrieved_raw_action_exact_count": len(exact_raw_actions),
                        "retrieved_raw_action_exact_return": (
                            exact_raw_actions[0]["discounted_return"]
                            if exact_raw_actions
                            else 0.0
                        ),
                        "retrieved_raw_action_exact_similarity": (
                            exact_raw_actions[0]["similarity"]
                            if exact_raw_actions
                            else 0.0
                        ),
                    }
                )
                used_exact.add(exact)
                added_for_pattern += 1

                if added_for_pattern >= max_per_canonical:
                    break
                if len(augmented_actions) >= max_actions:
                    break

            if len(augmented_actions) >= max_actions:
                break

        return {
            "retrieval": retrieval_result,
            "num_positive_patterns": len(positive_patterns),
            "positive_patterns": positive_patterns[:10],
            "augmented_actions": augmented_actions,
        }

    # ------------------------------------------------------------------
    # Action normalization and similarity computation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_action(action: Any) -> str:
        """Normalize an ALFWorld command for exact memory grouping.

        This normalization is intentionally simple: lower-casing and
        collapsing whitespace preserve action semantics while making
        equivalent formatting variants match.
        """

        text = "" if action is None else str(action)
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def canonicalize_action(cls, action: Any) -> str:
        """Canonicalize ALFWorld action by removing object instance ids.

        Examples:
            take apple 1 from countertop 2
            -> take apple <id> from countertop <id>

            put lettuce 2 in fridge 1
            -> put lettuce <id> in fridge <id>
        """
        normalized = cls.normalize_action(action)
        if not normalized:
            return ""

        tokens = normalized.split()
        canonical_tokens = [
            "<id>" if token.isdigit() else token
            for token in tokens
        ]
        return " ".join(canonical_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text in the same lightweight style as JitRL ALFWorld."""

        return [
            token
            for token in (text or "").lower().replace("\n", " ").split()
            if token.isalpha() or token.isalnum()
        ]

    @staticmethod
    def _get_ngrams(
        tokens: Sequence[str],
        ngram: int,
    ) -> List[str]:
        """Construct contiguous token n-grams."""

        if ngram <= 0:
            raise ValueError("ngram must be a positive integer.")

        if len(tokens) < ngram:
            return list(tokens)

        return [
            " ".join(tokens[index : index + ngram])
            for index in range(len(tokens) - ngram + 1)
        ]

    @classmethod
    def _multiset_jaccard(
        cls,
        left_tokens: Sequence[str],
        right_tokens: Sequence[str],
        *,
        ngram: int,
    ) -> float:
        """Compute frequency-aware Jaccard similarity over n-grams."""

        left_counter = Counter(cls._get_ngrams(left_tokens, ngram))
        right_counter = Counter(cls._get_ngrams(right_tokens, ngram))

        if not left_counter or not right_counter:
            return 0.0

        intersection = sum(
            (left_counter & right_counter).values()
        )
        union = sum(
            (left_counter | right_counter).values()
        )

        return intersection / union if union > 0 else 0.0

    def _effective_threshold(
        self,
        *,
        num_previous_actions: int,
    ) -> float:
        """Return the current retrieval threshold.

        With dynamic threshold enabled, the threshold decreases linearly by at
        most ``threshold_decay`` during the first
        ``threshold_decay_horizon`` executed actions.
        """

        if not self.dynamic_threshold:
            return self.similarity_threshold

        progress = min(
            1.0,
            num_previous_actions / float(self.threshold_decay_horizon),
        )

        return max(
            0.0,
            self.similarity_threshold - self.threshold_decay * progress,
        )

    # ------------------------------------------------------------------
    # Advantage normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_advantages_like_jitrl(
        raw_advantages: Mapping[str, float],
    ) -> Dict[str, float]:
        """Normalize advantages following the current JitRL implementation.

        If at least one retrieved action has positive advantage, every
        advantage is divided by the largest positive advantage.

        Otherwise, advantages are divided by the absolute largest negative
        advantage.

        This reproduces the current JitRL behavior and intentionally does not
        clip large negative normalized values.
        """

        if not raw_advantages:
            return {}

        values = list(raw_advantages.values())
        positive_values = [
            value for value in values if value > 0.0
        ]

        if positive_values:
            denominator = max(positive_values)
        else:
            denominator = abs(min(values))

        if denominator <= 0.0:
            return {
                action: 0.0
                for action in raw_advantages
            }

        return {
            action: advantage / denominator
            for action, advantage in raw_advantages.items()
        }

    @staticmethod
    def _unsupported_candidate_score(
        *,
        input_action: str,
        normalized_action: str,
        canonical_action: str = "",
    ) -> Dict[str, Any]:
        """Return neutral memory correction for an unsupported candidate."""

        return {
        "input_action": input_action,
        "normalized_action": normalized_action,
        "canonical_action": canonical_action,
        "has_memory_support": False,
        "match_count": 0,
        "mean_return": 0.0,
        "raw_advantage": 0.0,
        "normalized_advantage": 0.0,
    }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stored_step_from_dict(
        item: Mapping[str, Any],
    ) -> StoredStep:
        """Deserialize one stored transition from JSONL."""

        return StoredStep(
            episode_number=int(item["episode_number"]),
            task_id=item.get("task_id"),
            attempt_id=item.get("attempt_id"),
            step_index=int(item["step_index"]),
            task_text=str(item.get("task_text", "")),
            observation=str(item.get("observation", "")),
            inventory=str(item.get("inventory", "")),
            previous_actions=list(item.get("previous_actions", [])),
            action=str(item.get("action", "")),
            raw_action=str(
                item.get("raw_action", item.get("action", ""))
            ),
            env_reward=float(item.get("env_reward", 0.0) or 0.0),
            episode_success=bool(
                item.get("episode_success", False)
            ),
            episode_final_reward=float(
                item.get("episode_final_reward", 0.0) or 0.0
            ),
            trajectory_context=str(
                item.get("trajectory_context", "")
            ),
            current_env_info=str(
                item.get("current_env_info", "")
            ),
            future_rewards=[
                float(reward or 0.0)
                for reward in item.get("future_rewards", [])
            ],
        )

    @staticmethod
    def _append_jsonl(
        path: str,
        item: Mapping[str, Any],
    ) -> None:
        """Append one JSON-serializable record to a JSONL file."""

        with open(path, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(dict(item), ensure_ascii=False) + "\n"
            )

    @staticmethod
    def _load_jsonl(
        path: str,
    ) -> List[Dict[str, Any]]:
        """Load a JSONL file; return an empty list if it does not exist."""

        if not os.path.exists(path):
            return []

        items: List[Dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in memory file {path!r} "
                        f"at line {line_number}."
                    ) from exc

        return items

    # ------------------------------------------------------------------
    # Basic validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_text(value: Any) -> str:
        """Convert any optional value to a stripped string."""

        return "" if value is None else str(value).strip()

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert a value to a finite float, falling back to zero."""

        try:
            result = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

        return result if math.isfinite(result) else 0.0

    @staticmethod
    def _validate_probability(
        name: str,
        value: float,
    ) -> float:
        """Validate a finite numeric value in the interval [0, 1]."""

        value = float(value)

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be a finite number in [0, 1], "
                f"got {value!r}."
            )

        return value

    @staticmethod
    def _validate_non_negative_float(
        name: str,
        value: float,
    ) -> float:
        """Validate a finite non-negative floating point value."""

        value = float(value)

        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be a finite non-negative number, "
                f"got {value!r}."
            )

        return value

    @staticmethod
    def _validate_positive_int(
        name: str,
        value: int,
    ) -> int:
        """Validate a positive integer parameter."""

        value = int(value)

        if value <= 0:
            raise ValueError(
                f"{name} must be positive, got {value!r}."
            )

        return value
