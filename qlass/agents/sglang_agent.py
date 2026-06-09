import json
import time
import logging
import re
from typing import Any, Dict, List

import requests
import os
import atexit
import transformers
from qlass.data_utils import get_chat_template

from .base import LMAgent

logger = logging.getLogger("agent_frame")

_ACTION_CANDIDATE_PROMPT_SUFFIX = (
    "\n\nGenerate exactly one reasonable next action for the current state. "
    "Follow the same response format as in the examples and previous assistant turns:\n"
    "Action: <action>\n"
    "Do not include explanations, analysis, numbering, or multiple actions."
)


class SGLangAgent(LMAgent):

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.server_address = config["server_address"]
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", -1)
        self.get_choice_rank = config.get("get_choice_rank", 0)
        self.num_samples = config.get("num_samples", 1)

        # Optional request settings for non-streaming candidate generation.
        self.request_timeout = config.get("request_timeout", None)
        self.stop_words = config.get("stop_words", None)

        # Optional prompt trimming (disabled by default)
        self.max_prompt_tokens = config.get("max_prompt_tokens", None)  # e.g. 3800
        self.keep_first_n = int(config.get("keep_first_n", 3))  # keep first 3 messages by default
        self.tokenizer_path = config.get("tokenizer_path", None)  # optional explicit path
        self._tokenizer = None

        self._trim_total_calls = 0
        self._trim_calls = 0
        self._trim_tokens_before_sum = 0
        self._trim_tokens_after_sum = 0
        self._trim_removed_msgs_sum = 0
        self._trim_log_every = int(config.get("trim_log_every", 200))  # как часто логировать статистику

        atexit.register(self._log_trim_summary)


    def _resolve_tokenizer_path(self) -> str:
        if self.tokenizer_path:
            return self.tokenizer_path
        # If model_name is already a path, use it
        if os.path.exists(self.model_name):
            return self.model_name
        # Try MODEL_PATH + model_name
        root = os.environ.get("MODEL_PATH", "")
        if root:
            cand = os.path.join(root, self.model_name)
            if os.path.exists(cand):
                return cand
        # Fallback: treat model_name as HF repo id
        return self.model_name

    def _get_tokenizer(self):
        if self._tokenizer is None:
            path = self._resolve_tokenizer_path()
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=False)
        return self._tokenizer

    def _build_prompt(self, chat, messages: List[dict]) -> str:
        # Same format as before: add assistant stub
        return chat.get_prompt(messages + [{"role": "assistant", "content": None}])


    def _build_action_candidate_messages(self, messages: List[dict]) -> List[dict]:
        """Return a copied message history with a canonical action-generation suffix.

        The original message history must not be modified in-place, because it is also
        used by the environment state, replay logic, and QNet evaluation.
        """
        if not messages:
            raise ValueError("[SGLangAgent] Cannot build candidate prompt from empty messages.")

        candidate_messages = [dict(message) for message in messages]

        if candidate_messages[-1].get("role") == "user":
            old_content = candidate_messages[-1].get("content") or ""
            candidate_messages[-1]["content"] = old_content.rstrip() + _ACTION_CANDIDATE_PROMPT_SUFFIX
        else:
            # Normally ALFWorld state history should end with a user observation.
            # Keep this fallback to make the method robust to unusual histories.
            candidate_messages.append(
                {
                    "role": "user",
                    "content": _ACTION_CANDIDATE_PROMPT_SUFFIX.strip(),
                }
            )

        return candidate_messages


    def _response_to_generation_items(self, response_json: Any) -> List[Dict[str, Any]]:
        """Normalize possible SGLang /generate response shapes to a list of items.

        Expected common shapes:
        1. [{"text": ..., "meta_info": ...}, ...]
        2. {"text": [...], "meta_info": [...]}
        3. {"text": "...", "meta_info": {...}}

        We keep this normalization defensive because SGLang response shape may differ
        between single, batched, and multi-sample calls.
        """
        if isinstance(response_json, list):
            return response_json

        if not isinstance(response_json, dict):
            raise RuntimeError(
                f"[SGLangAgent] Unexpected SGLang response type: {type(response_json)}"
            )

        text = response_json.get("text")
        meta_info = response_json.get("meta_info")

        if isinstance(text, list):
            if not isinstance(meta_info, list):
                raise RuntimeError(
                    "[SGLangAgent] SGLang returned text as a list, but meta_info is not a list. "
                    f"Cannot align texts with logprobs. response_keys={list(response_json.keys())}"
                )

            if len(text) != len(meta_info):
                raise RuntimeError(
                    "[SGLangAgent] SGLang returned different number of texts and meta_info entries: "
                    f"len(text)={len(text)}, len(meta_info)={len(meta_info)}"
                )

            return [
                {
                    "text": cur_text,
                    "meta_info": cur_meta,
                }
                for cur_text, cur_meta in zip(text, meta_info)
            ]

        return [response_json]


    def _extract_action_text(self, text: str) -> str:
        """Extract the raw action after 'Action:' if present.

        The environment can parse the full model output, but for deduplication and
        logging it is much cleaner to store the parsed action separately.
        """
        stripped = text.strip()

        match = re.search(r"Action:\s*(.*)", stripped, flags=re.IGNORECASE | re.DOTALL)
        if match:
            action = match.group(1).strip()
        else:
            action = stripped

        # If the model accidentally continues after the action, keep only the first line.
        action = action.splitlines()[0].strip()
        return action


    def _candidate_dedup_key(self, text: str) -> str:
        """Normalize generated action for deduplication."""
        action_text = self._extract_action_text(text)
        action_text = action_text.strip().lower()
        action_text = action_text.rstrip(".")
        return " ".join(action_text.split())


    def _extract_output_logprob_stats(self, item: Dict[str, Any]) -> Dict[str, float]:
        """Extract sum/mean output-token logprob from one SGLang generation item.

        We intentionally fail loudly if output-token logprobs are missing: without
        them the new algorithm degenerates into ordinary QLASS-style candidate scoring.
        """
        meta_info = item.get("meta_info") or {}
        output_token_logprobs = meta_info.get("output_token_logprobs")

        if not output_token_logprobs:
            raise RuntimeError(
                "[SGLangAgent] Missing meta_info.output_token_logprobs in SGLang response. "
                "This method requires return_logprob=True and logprob_start_len=-1."
            )

        logprobs: List[float] = []

        for pos, entry in enumerate(output_token_logprobs):
            # SGLang commonly returns entries like:
            #   [logprob, token_id, extra]
            # where extra can be None or token text depending on return_text_in_logprobs.
            if isinstance(entry, (list, tuple)):
                if len(entry) == 0:
                    raise RuntimeError(
                        f"[SGLangAgent] Empty output_token_logprobs entry at position {pos}."
                    )
                token_logprob = entry[0]

            elif isinstance(entry, dict):
                token_logprob = entry.get("logprob", entry.get("token_logprob"))

            else:
                raise RuntimeError(
                    "[SGLangAgent] Unsupported output_token_logprobs entry type "
                    f"at position {pos}: {type(entry)}. entry={entry}"
                )

            if token_logprob is None:
                raise RuntimeError(
                    f"[SGLangAgent] Missing token logprob at output position {pos}. "
                    f"entry={entry}"
                )

            logprobs.append(float(token_logprob))

        if not logprobs:
            raise RuntimeError("[SGLangAgent] No output-token logprobs were extracted.")

        logprob_sum = float(sum(logprobs))
        logprob_mean = float(logprob_sum / len(logprobs))

        return {
            "actor_logprob_sum": logprob_sum,
            "actor_logprob_mean": logprob_mean,
            "num_output_tokens": len(logprobs),
        }


    def _maybe_trim_messages(self, chat, messages: List[dict]) -> List[dict]:
        if not self.max_prompt_tokens:
            return messages

        tok = self._get_tokenizer()
        prompt = self._build_prompt(chat, messages)
        n_tokens = len(tok(prompt, add_special_tokens=True).input_ids)
        if n_tokens <= self.max_prompt_tokens:
            return messages

        # Keep the first N messages unchanged, trim oldest from the remainder
        keep_n = min(self.keep_first_n, len(messages))
        prefix = messages[:keep_n]
        rest = messages[keep_n:]

        before = n_tokens
        removed = 0
        # Remove oldest messages until it fits (one-by-one to be safe)
        while rest and n_tokens > self.max_prompt_tokens:
            rest.pop(0)
            removed += 1
            prompt = self._build_prompt(chat, prefix + rest)
            n_tokens = len(tok(prompt, add_special_tokens=True).input_ids)

        if removed > 0:
            # logger.warning(
            #     f"[SGLangAgent] Prompt trimmed: {before} -> {n_tokens} tokens "
            #     f"(removed {removed} msgs, kept_first_n={keep_n})"
            # )

            self._trim_calls += 1
            self._trim_tokens_before_sum += before
            self._trim_tokens_after_sum += n_tokens
            self._trim_removed_msgs_sum += removed

            # периодический лог каждые trim_log_every вызовов агента
            if self._trim_log_every > 0 and (self._trim_total_calls % self._trim_log_every == 0):
                ratio = self._trim_calls / max(1, self._trim_total_calls)
                logger.warning(
                    f"[TRIM_STATS] pid={os.getpid()} total_calls={self._trim_total_calls} "
                    f"trim_calls={self._trim_calls} ratio={ratio:.3f} "
                    f"avg_before={self._trim_tokens_before_sum/max(1,self._trim_calls):.1f} "
                    f"avg_after={self._trim_tokens_after_sum/max(1,self._trim_calls):.1f} "
                    f"avg_removed_msgs={self._trim_removed_msgs_sum/max(1,self._trim_calls):.2f}"
                )
        return prefix + rest


    def _log_trim_summary(self):
        if self._trim_total_calls <= 0:
            return
        ratio = self._trim_calls / max(1, self._trim_total_calls)
        logger.warning(
            f"[TRIM_SUMMARY] pid={os.getpid()} total_calls={self._trim_total_calls} "
            f"trim_calls={self._trim_calls} ratio={ratio:.3f} "
            f"avg_before={self._trim_tokens_before_sum/max(1,self._trim_calls):.1f} "
            f"avg_after={self._trim_tokens_after_sum/max(1,self._trim_calls):.1f} "
            f"avg_removed_msgs={self._trim_removed_msgs_sum/max(1,self._trim_calls):.2f}"
        )


    def generate_candidates_with_logprobs(
        self,
        messages: List[dict],
        n: int,
        *,
        add_action_candidate_prompt: bool = False,
        oversample_factor: int = 3,
        max_rounds: int = 2,
    ) -> List[Dict[str, Any]]:
        """Generate up to n unique candidate actions from one canonical prompt.

        The method samples more raw completions than needed and deduplicates them by
        parsed action text. All raw completions are sampled from the same canonical
        prompt, so their actor logprobs are comparable.

        Returns:
            [
                {
                    "text": str,                 # full generated text, e.g. "Action: open fridge"
                    "action": str,               # parsed action, e.g. "open fridge"
                    "actor_logprob_sum": float,  # sum of output-token logprobs
                    "actor_logprob_mean": float, # mean output-token logprob; main score to use
                    "num_output_tokens": int,
                    "finish_reason": Any,
                    "raw_index": int,
                    "round_index": int,
                },
                ...
            ]

        Hard errors:
            - empty server address;
            - missing output-token logprobs;
            - malformed SGLang response;
            - no unique candidates generated.
        """
        if n <= 0:
            raise ValueError(f"[SGLangAgent] n must be positive, got n={n}.")
        if oversample_factor <= 0:
            raise ValueError(
                f"[SGLangAgent] oversample_factor must be positive, got {oversample_factor}."
            )
        if max_rounds <= 0:
            raise ValueError(f"[SGLangAgent] max_rounds must be positive, got {max_rounds}.")

        self._trim_total_calls += 1

        server_addr = self.server_address
        if server_addr == "":
            raise ValueError("[SGLangAgent] Empty SGLang server_address.")

        sampling_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            # We send a batch of identical prompts; each prompt should produce one sample.
            "n": 1,
        }

        if self.stop_words:
            sampling_params["stop"] = self.stop_words

        chat = get_chat_template(self.model_name)

        candidate_messages = (
            self._build_action_candidate_messages(messages)
            if add_action_candidate_prompt
            else [dict(message) for message in messages]
        )

        candidate_messages = self._maybe_trim_messages(chat, candidate_messages)
        prompt = self._build_prompt(chat, candidate_messages)

        prompt_tokens = None
        try:
            tokenizer = self._get_tokenizer()
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=True).input_ids)
        except Exception as exc:
            logger.warning(
                "[SGLangAgent] Failed to count prompt tokens for candidate generation: %s",
                exc,
            )

        target_n = n
        raw_n_per_round = max(target_n, target_n * oversample_factor)

        logger.info(
            "[SGLangAgent] Starting candidate generation with logprobs: "
            "target_n=%d, raw_n_per_round=%d, max_rounds=%d, "
            "temperature=%s, top_p=%s, top_k=%s, max_new_tokens=%s, prompt_tokens=%s",
            target_n,
            raw_n_per_round,
            max_rounds,
            self.temperature,
            self.top_p,
            self.top_k,
            self.max_new_tokens,
            prompt_tokens,
        )

        candidates: List[Dict[str, Any]] = []
        seen_keys = set()
        last_error = None

        for round_idx in range(max_rounds):
            payload = {
                # Batch of identical prompts: comparable actor logprobs under one policy context.
                "text": [prompt for _ in range(raw_n_per_round)],
                "sampling_params": sampling_params,
                "stream": False,
                "return_logprob": True,
                # -1 means: output-token logprobs only.
                "logprob_start_len": -1,
                "return_text_in_logprobs": False,
            }

            logger.info(
                "[SGLangAgent] Candidate generation round %d/%d: requesting %d raw samples.",
                round_idx + 1,
                max_rounds,
                raw_n_per_round,
            )

            try:
                response = requests.post(
                    server_addr + "/generate",
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                raw_items = self._response_to_generation_items(response.json())

                if len(raw_items) != raw_n_per_round:
                    logger.warning(
                        "[SGLangAgent] Expected %d raw generation items, got %d. "
                        "Continuing with returned items.",
                        raw_n_per_round,
                        len(raw_items),
                    )

                round_unique = 0
                round_duplicates = 0
                round_empty = 0

                for raw_index, item in enumerate(raw_items):
                    text = str(item.get("text", "")).strip()

                    if not text:
                        round_empty += 1
                        logger.warning(
                            "[SGLangAgent] Dropping empty candidate: round=%d raw_index=%d",
                            round_idx + 1,
                            raw_index,
                        )
                        continue

                    # Hard error if logprobs are absent or malformed.
                    logprob_stats = self._extract_output_logprob_stats(item)

                    action = self._extract_action_text(text)
                    dedup_key = self._candidate_dedup_key(text)

                    if not dedup_key:
                        round_empty += 1
                        logger.warning(
                            "[SGLangAgent] Dropping candidate with empty dedup key: "
                            "round=%d raw_index=%d text=%r",
                            round_idx + 1,
                            raw_index,
                            text,
                        )
                        continue

                    if dedup_key in seen_keys:
                        round_duplicates += 1
                        logger.debug(
                            "[SGLangAgent] Dropping duplicate candidate: "
                            "round=%d raw_index=%d action=%r text=%r",
                            round_idx + 1,
                            raw_index,
                            action,
                            text,
                        )
                        continue

                    seen_keys.add(dedup_key)

                    meta_info = item.get("meta_info") or {}
                    candidate = {
                        "text": text,
                        "action": action,
                        "actor_logprob_sum": logprob_stats["actor_logprob_sum"],
                        "actor_logprob_mean": logprob_stats["actor_logprob_mean"],
                        "num_output_tokens": int(logprob_stats["num_output_tokens"]),
                        "finish_reason": meta_info.get("finish_reason"),
                        "raw_index": raw_index,
                        "round_index": round_idx,
                    }
                    candidates.append(candidate)
                    round_unique += 1

                    logger.debug(
                        "[SGLangAgent] Accepted candidate %d/%d: "
                        "round=%d raw_index=%d action=%r text=%r "
                        "logprob_mean=%.6f logprob_sum=%.6f num_tokens=%d finish_reason=%s",
                        len(candidates),
                        target_n,
                        round_idx + 1,
                        raw_index,
                        candidate["action"],
                        candidate["text"],
                        candidate["actor_logprob_mean"],
                        candidate["actor_logprob_sum"],
                        candidate["num_output_tokens"],
                        candidate["finish_reason"],
                    )

                    if len(candidates) >= target_n:
                        logger.info(
                            "[SGLangAgent] Generated %d unique candidates after %d/%d rounds. "
                            "Last round stats: unique=%d, duplicates=%d, empty=%d.",
                            len(candidates),
                            round_idx + 1,
                            max_rounds,
                            round_unique,
                            round_duplicates,
                            round_empty,
                        )
                        return candidates[:target_n]

                logger.info(
                    "[SGLangAgent] Round %d/%d finished: total_unique=%d/%d, "
                    "round_unique=%d, round_duplicates=%d, round_empty=%d.",
                    round_idx + 1,
                    max_rounds,
                    len(candidates),
                    target_n,
                    round_unique,
                    round_duplicates,
                    round_empty,
                )

            except Exception as exc:
                last_error = exc
                logger.exception(
                    "[SGLangAgent] Candidate generation round %d/%d failed.",
                    round_idx + 1,
                    max_rounds,
                )
                # If logprobs are missing, retrying usually will not help, but we still keep
                # the round-level retry semantics because transient server failures are possible.
                if round_idx + 1 < max_rounds:
                    time.sleep(5)

        if not candidates:
            raise RuntimeError(
                "[SGLangAgent] Failed to generate any unique candidates with logprobs "
                f"after {max_rounds} rounds."
            ) from last_error

        logger.warning(
            "[SGLangAgent] Requested %d unique candidates, but got only %d after %d rounds. "
            "Proceeding with fewer candidates.",
            target_n,
            len(candidates),
            max_rounds,
        )

        return candidates


    def __call__(self, messages: List[dict],get_choice_rank:int=0) -> str:
        # import ipdb; ipdb.set_trace()
        # if get_choice_rank != -1:

        self._trim_total_calls += 1

        self.get_choice_rank = get_choice_rank
        server_addr = self.server_address
        if server_addr == "":
            raise ValueError
        sampling_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n": self.num_samples,
        }
        chat = get_chat_template(self.model_name)
        # prompt = chat.get_prompt(messages + [{"role": "assistant", "content": None}])
        messages = self._maybe_trim_messages(chat, messages)
        prompt = self._build_prompt(chat, messages)
        for _ in range(3):
            # import ipdb; ipdb.set_trace()
            try:
                response = requests.post(
                    server_addr + "/generate",
                    json = {
                        "text": prompt,
                        "sampling_params": sampling_params,
                        "stream": True,
                    },
                )
                text = ""
                for chunk in response.iter_lines(decode_unicode=False):
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        text = data["text"].strip()
                return text

            except Exception as e:
                print(f"Error when agent try to continue the conversation: {e},retrying...")
            time.sleep(5)
        return ""
