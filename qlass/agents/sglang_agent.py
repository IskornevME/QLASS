import json
import time
import logging
from typing import List
import requests
import os
import atexit
import transformers
from qlass.data_utils import get_chat_template

from .base import LMAgent

logger = logging.getLogger("agent_frame")


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

        # Optional prompt trimming (disabled by default)
        self.max_prompt_tokens = config.get("max_prompt_tokens", None)  # e.g. 3800
        self.keep_first_n = int(config.get("keep_first_n", 3))  # keep first 3 messages by default
        self.min_tail_msgs = int(
            config.get("min_tail_msgs", 4)
        )
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

    def _maybe_trim_messages(
        self,
        chat,
        messages: List[dict],
    ) -> List[dict]:
        if not self.max_prompt_tokens:
            return messages

        tok = self._get_tokenizer()

        def count_tokens(
            current_messages: List[dict],
        ) -> int:
            prompt = self._build_prompt(
                chat,
                current_messages,
            )
            return len(
                tok(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                ).input_ids
            )

        n_tokens_before = count_tokens(
            messages
        )

        if (
            n_tokens_before
            <= self.max_prompt_tokens
        ):
            return messages

        keep_n = min(
            self.keep_first_n,
            len(messages),
        )

        prefix = messages[:keep_n]
        rest = messages[keep_n:]

        n_tokens = n_tokens_before
        removed = 0
        relaxed_tail_constraint = False

        # Первый этап:
        # удаляем старые полные interaction turns,
        # но сохраняем как минимум min_tail_msgs.
        preferred_min_tail = max(
            int(self.min_tail_msgs),
            2,
        )

        while (
            n_tokens > self.max_prompt_tokens
            and len(rest) - preferred_min_tail >= 2
        ):
            rest = rest[2:]
            removed += 2

            n_tokens = count_tokens(
                prefix + rest
            )

        # Второй, аварийный этап:
        # policy вызывается на истории, которая заканчивается
        # последним user observation. Поэтому минимально нужно
        # сохранить два последних сообщения:
        #
        #   assistant: последнее действие
        #   user: последнее наблюдение
        min_safe_tail = 2

        while (
            n_tokens > self.max_prompt_tokens
            and len(rest) - min_safe_tail >= 2
        ):
            relaxed_tail_constraint = True

            rest = rest[2:]
            removed += 2

            n_tokens = count_tokens(
                prefix + rest
            )

        trimmed_messages = prefix + rest

        if relaxed_tail_constraint:
            logger.warning(
                "[POLICY_TRIM_RELAXED_TAIL] "
                "The configured minimum tail could not be preserved. "
                "before_tokens=%d after_tokens=%d "
                "configured_min_tail=%d actual_tail=%d "
                "removed_messages=%d",
                n_tokens_before,
                n_tokens,
                self.min_tail_msgs,
                len(rest),
                removed,
            )

        if n_tokens > self.max_prompt_tokens:
            raise RuntimeError(
                "[POLICY_TRIM_FAILED] "
                "Policy prompt cannot be reduced to the configured "
                "limit while preserving the initial prefix and the "
                "latest action/observation pair. "
                f"tokens={n_tokens}, "
                f"limit={self.max_prompt_tokens}, "
                f"prefix_messages={len(prefix)}, "
                f"tail_messages={len(rest)}"
            )

        self._trim_calls += 1
        self._trim_tokens_before_sum += (
            n_tokens_before
        )
        self._trim_tokens_after_sum += (
            n_tokens
        )
        self._trim_removed_msgs_sum += removed

        logger.info(
            "[POLICY_TRIM] "
            "before_tokens=%d after_tokens=%d "
            "removed_messages=%d prefix=%d tail=%d",
            n_tokens_before,
            n_tokens,
            removed,
            len(prefix),
            len(rest),
        )

        return trimmed_messages


    def _log_trim_summary(self) -> None:
        if self._trim_total_calls <= 0:
            return

        trim_ratio = (
            self._trim_calls
            / self._trim_total_calls
        )

        logger.warning(
            "[TRIM_SUMMARY] "
            "pid=%d "
            "total_calls=%d "
            "trim_calls=%d "
            "ratio=%.3f "
            "avg_before=%.1f "
            "avg_after=%.1f "
            "avg_removed_msgs=%.2f",
            os.getpid(),
            self._trim_total_calls,
            self._trim_calls,
            trim_ratio,
            (
                self._trim_tokens_before_sum
                / max(1, self._trim_calls)
            ),
            (
                self._trim_tokens_after_sum
                / max(1, self._trim_calls)
            ),
            (
                self._trim_removed_msgs_sum
                / max(1, self._trim_calls)
            ),
        )


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
