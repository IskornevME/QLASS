"""Generic SGLang chat-completions agent.

Unlike ``SGLangAgent``, this class does not build a model-specific prompt
manually. It sends structured chat messages to SGLang's OpenAI-compatible
``/v1/chat/completions`` endpoint, so the server applies the chat template
shipped with the served model.

The class is intended for general instruction-tuned models such as
Qwen3-4B-Instruct-2507, while preserving the message-level prompt trimming
used by the existing QLASS policy agent.
"""

from __future__ import annotations

import atexit
import copy
import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional

import requests
import transformers

from .base import LMAgent


logger = logging.getLogger("agent_frame")


class SGLangChatAgent(LMAgent):
    """Policy/critic client for an OpenAI-compatible SGLang server."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)

        self.server_address = str(config["server_address"]).rstrip("/")
        if not self.server_address:
            raise ValueError(
                "SGLangChatAgent requires a non-empty server_address."
            )

        self.model_name = str(config["model_name"]).strip()
        if not self.model_name:
            raise ValueError(
                "SGLangChatAgent requires a non-empty model_name."
            )

        # Identifier sent in the OpenAI-compatible request. If omitted, it is
        # resolved lazily through GET /v1/models.
        self.api_model_name = str(
            config.get("api_model_name", "")
            or config.get("served_model_name", "")
        ).strip()
        self._resolved_api_model_name: Optional[str] = None

        self.tokenizer_path = config.get("tokenizer_path")
        self._tokenizer = None

        self.temperature = float(config.get("temperature", 0.0))
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.top_p = float(config.get("top_p", 1.0))

        top_k = config.get("top_k", None)
        self.top_k = None if top_k is None else int(top_k)

        min_p = config.get("min_p", None)
        self.min_p = None if min_p is None else float(min_p)

        self.presence_penalty = float(config.get("presence_penalty", 0.0))
        self.frequency_penalty = float(config.get("frequency_penalty", 0.0))

        repetition_penalty = config.get("repetition_penalty", None)
        self.repetition_penalty = (
            None if repetition_penalty is None else float(repetition_penalty)
        )

        seed = config.get("seed", None)
        self.seed = None if seed is None else int(seed)

        self.num_samples = int(config.get("num_samples", 1))
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        self.get_choice_rank = int(config.get("get_choice_rank", 0))

        self.max_prompt_tokens = config.get("max_prompt_tokens", None)
        if self.max_prompt_tokens is not None:
            self.max_prompt_tokens = int(self.max_prompt_tokens)
            if self.max_prompt_tokens <= 0:
                raise ValueError(
                    "max_prompt_tokens must be positive or None."
                )

        self.keep_first_n = int(config.get("keep_first_n", 3))
        self.min_tail_msgs = int(config.get("min_tail_msgs", 4))
        if self.keep_first_n < 0:
            raise ValueError("keep_first_n must be non-negative.")
        if self.min_tail_msgs < 0:
            raise ValueError("min_tail_msgs must be non-negative.")

        self.request_timeout = float(config.get("request_timeout", 300.0))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_delay_seconds = float(
            config.get("retry_delay_seconds", 5.0)
        )
        if self.request_timeout <= 0.0:
            raise ValueError("request_timeout must be positive.")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive.")
        if self.retry_delay_seconds < 0.0:
            raise ValueError("retry_delay_seconds must be non-negative.")

        self.response_format_reminder = str(
            config.get("response_format_reminder", "")
        ).strip()

        self.chat_template_kwargs = dict(
            config.get("chat_template_kwargs", {}) or {}
        )
        forbidden_template_kwargs = {
            "tokenize",
            "add_generation_prompt",
            "return_tensors",
            "return_dict",
            "truncation",
        }
        invalid_template_kwargs = (
            forbidden_template_kwargs & self.chat_template_kwargs.keys()
        )
        if invalid_template_kwargs:
            raise ValueError(
                "chat_template_kwargs contains arguments controlled "
                "internally by SGLangChatAgent: "
                f"{sorted(invalid_template_kwargs)}"
            )

        self.extra_request_body = dict(
            config.get("extra_request_body", {}) or {}
        )

        stop = config.get("stop", None)
        if stop is None and bool(config.get("use_default_stop_words", False)):
            stop = list(self.stop_words)

        if isinstance(stop, str):
            self.stop: Optional[List[str]] = [stop]
        elif stop is None:
            self.stop = None
        else:
            self.stop = [str(item) for item in stop if str(item)]

        self.api_key = str(
            config.get(
                "api_key",
                os.environ.get("OPENAI_API_KEY", ""),
            )
            or ""
        )

        self._session = requests.Session()

        self._trim_total_calls = 0
        self._trim_calls = 0
        self._trim_tokens_before_sum = 0
        self._trim_tokens_after_sum = 0
        self._trim_removed_msgs_sum = 0

        atexit.register(self._log_trim_summary)

    # ------------------------------------------------------------------
    # Tokenizer and message preparation
    # ------------------------------------------------------------------

    def _resolve_tokenizer_path(self) -> str:
        if self.tokenizer_path:
            return str(self.tokenizer_path)

        if os.path.exists(self.model_name):
            return self.model_name

        model_root = os.environ.get("MODEL_PATH", "")
        if model_root:
            candidate = os.path.join(model_root, self.model_name)
            if os.path.exists(candidate):
                return candidate

        # This also permits a Hugging Face model id.
        return self.model_name

    def _get_tokenizer(self):
        if self._tokenizer is None:
            tokenizer_path = self._resolve_tokenizer_path()
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=False,
            )

            if not getattr(self._tokenizer, "chat_template", None):
                raise ValueError(
                    "The tokenizer does not define a chat template: "
                    f"{tokenizer_path!r}. Use a chat/instruct checkpoint "
                    "with a valid tokenizer.chat_template."
                )

        return self._tokenizer

    @staticmethod
    def _normalize_messages(messages: List[dict]) -> List[Dict[str, Any]]:
        if not messages:
            raise ValueError(
                "SGLangChatAgent received an empty message list."
            )

        normalized: List[Dict[str, Any]] = []

        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(
                    "Every chat message must be a mapping; "
                    f"message {index} has type {type(message).__name__}."
                )

            role = str(message.get("role", "")).strip()
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Unsupported role at message {index}: {role!r}."
                )

            content = message.get("content", "")
            if content is None:
                content = ""

            normalized_message: Dict[str, Any] = {
                "role": role,
                "content": str(content),
            }

            # Preserve an optional OpenAI-compatible name field.
            if message.get("name") is not None:
                normalized_message["name"] = str(message["name"])

            normalized.append(normalized_message)

        return normalized

    def _append_response_format_reminder(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.response_format_reminder:
            return messages

        result = copy.deepcopy(messages)

        for index in range(len(result) - 1, -1, -1):
            if result[index]["role"] != "user":
                continue

            original_content = str(
                result[index].get("content", "")
            ).rstrip()

            if original_content:
                result[index]["content"] = (
                    original_content
                    + "\n\n"
                    + self.response_format_reminder
                )
            else:
                result[index]["content"] = self.response_format_reminder

            return result

        raise ValueError(
            "response_format_reminder is configured, but the conversation "
            "contains no user message."
        )

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        tokenizer = self._get_tokenizer()

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            **self.chat_template_kwargs,
        )

        return len(token_ids)

    def _maybe_trim_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.max_prompt_tokens is None:
            return messages

        n_tokens_before = self._count_tokens(messages)

        if n_tokens_before <= self.max_prompt_tokens:
            return messages

        keep_n = min(self.keep_first_n, len(messages))
        prefix = list(messages[:keep_n])
        rest = list(messages[keep_n:])

        n_tokens = n_tokens_before
        removed = 0
        relaxed_tail_constraint = False

        # First preserve the configured number of recent messages. Removing
        # two adjacent messages keeps the role parity unchanged.
        preferred_min_tail = max(self.min_tail_msgs, 2)

        while (
            n_tokens > self.max_prompt_tokens
            and len(rest) - preferred_min_tail >= 2
        ):
            rest = rest[2:]
            removed += 2
            n_tokens = self._count_tokens(prefix + rest)

        # Emergency stage: retain at least the most recent
        # assistant-action/user-observation pair.
        min_safe_tail = 2

        while (
            n_tokens > self.max_prompt_tokens
            and len(rest) - min_safe_tail >= 2
        ):
            relaxed_tail_constraint = True
            rest = rest[2:]
            removed += 2
            n_tokens = self._count_tokens(prefix + rest)

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
                "The chat prompt cannot be reduced to the configured limit "
                "while preserving the initial prefix and the latest "
                "action/observation context. "
                f"tokens={n_tokens}, limit={self.max_prompt_tokens}, "
                f"prefix_messages={len(prefix)}, tail_messages={len(rest)}"
            )

        self._trim_calls += 1
        self._trim_tokens_before_sum += n_tokens_before
        self._trim_tokens_after_sum += n_tokens
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

        trim_ratio = self._trim_calls / self._trim_total_calls

        logger.warning(
            "[TRIM_SUMMARY] "
            "pid=%d total_calls=%d trim_calls=%d ratio=%.3f "
            "avg_before=%.1f avg_after=%.1f avg_removed_msgs=%.2f",
            os.getpid(),
            self._trim_total_calls,
            self._trim_calls,
            trim_ratio,
            self._trim_tokens_before_sum / max(1, self._trim_calls),
            self._trim_tokens_after_sum / max(1, self._trim_calls),
            self._trim_removed_msgs_sum / max(1, self._trim_calls),
        )

    # ------------------------------------------------------------------
    # OpenAI-compatible SGLang request
    # ------------------------------------------------------------------

    def _request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _resolve_api_model_name(self) -> str:
        if self._resolved_api_model_name:
            return self._resolved_api_model_name

        if self.api_model_name:
            self._resolved_api_model_name = self.api_model_name
            return self._resolved_api_model_name

        fallback = self.model_name

        try:
            response = self._session.get(
                self.server_address + "/v1/models",
                headers=self._request_headers(),
                timeout=min(self.request_timeout, 30.0),
            )
            response.raise_for_status()
            payload = response.json()

            model_ids = [
                str(item.get("id", "")).strip()
                for item in payload.get("data", [])
                if str(item.get("id", "")).strip()
            ]

            if self.model_name in model_ids:
                resolved = self.model_name
            else:
                model_basename = os.path.basename(
                    self.model_name.rstrip("/")
                )
                basename_matches = [
                    model_id
                    for model_id in model_ids
                    if os.path.basename(model_id.rstrip("/"))
                    == model_basename
                ]

                if len(basename_matches) == 1:
                    resolved = basename_matches[0]
                elif len(model_ids) == 1:
                    resolved = model_ids[0]
                else:
                    resolved = fallback
                    if model_ids:
                        logger.warning(
                            "[SGLANG_MODEL_ID_AMBIGUOUS] "
                            "Configured model_name=%r, server models=%r; "
                            "using configured value.",
                            self.model_name,
                            model_ids,
                        )

        except Exception as exc:
            logger.warning(
                "[SGLANG_MODEL_ID_FALLBACK] "
                "Could not resolve the served model through /v1/models: "
                "%s; using configured model_name=%r.",
                exc,
                fallback,
            )
            resolved = fallback

        self._resolved_api_model_name = resolved
        return resolved

    def _build_request_body(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._resolve_api_model_name(),
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "n": self.num_samples,
            "stream": False,
        }

        # SGLang-supported OpenAI-compatible extensions. A negative top_k
        # means "not configured" and is therefore omitted.
        if self.top_k is not None and self.top_k >= 0:
            payload["top_k"] = self.top_k

        if self.min_p is not None:
            payload["min_p"] = self.min_p

        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty

        if self.seed is not None:
            payload["seed"] = self.seed

        if self.stop:
            payload["stop"] = self.stop

        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(
                self.chat_template_kwargs
            )

        protected_keys = {"model", "messages", "stream"}
        collisions = protected_keys & self.extra_request_body.keys()
        if collisions:
            raise ValueError(
                "extra_request_body may not override: "
                f"{sorted(collisions)}"
            )

        payload.update(self.extra_request_body)
        return payload

    @staticmethod
    def _extract_choice_text(
        response_payload: Mapping[str, Any],
        *,
        choice_rank: int,
    ) -> str:
        choices = response_payload.get("choices", [])
        if not isinstance(choices, list):
            raise ValueError(
                "SGLang response field 'choices' is not a list."
            )
        if not choices:
            raise ValueError("SGLang response contains no choices.")

        if not 0 <= choice_rank < len(choices):
            raise IndexError(
                f"Requested choice rank {choice_rank}, but the response "
                f"contains {len(choices)} choices."
            )

        choice = choices[choice_rank]
        if not isinstance(choice, Mapping):
            raise ValueError("SGLang response choice is not an object.")

        message = choice.get("message", {})
        if not isinstance(message, Mapping):
            raise ValueError(
                "SGLang response choice has no valid message object."
            )

        content = message.get("content", None)
        if content is None:
            reasoning_preview = str(
                message.get("reasoning_content", "") or ""
            )[:200]
            raise ValueError(
                "SGLang response message has content=None. "
                f"finish_reason={choice.get('finish_reason')!r} "
                f"matched_stop={choice.get('matched_stop')!r} "
                f"usage={response_payload.get('usage')!r} "
                f"reasoning_content_preview={reasoning_preview!r}"
            )

        text = str(content).strip()
        if not text:
            raise ValueError("SGLang returned an empty assistant response.")

        return text

    def __call__(
        self,
        messages: List[dict],
        get_choice_rank: int = 0,
    ) -> str:
        self._trim_total_calls += 1
        self.get_choice_rank = int(get_choice_rank)

        prepared_messages = self._normalize_messages(messages)
        prepared_messages = self._append_response_format_reminder(
            prepared_messages
        )
        prepared_messages = self._maybe_trim_messages(prepared_messages)

        payload = self._build_request_body(prepared_messages)

        last_error: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    self.server_address + "/v1/chat/completions",
                    headers=self._request_headers(),
                    json=payload,
                    timeout=self.request_timeout,
                )

                if not response.ok:
                    body_preview = response.text[:1000]
                    raise requests.HTTPError(
                        "SGLang chat request failed: "
                        f"status={response.status_code}, "
                        f"body={body_preview!r}",
                        response=response,
                    )

                response_payload = response.json()
                return self._extract_choice_text(
                    response_payload,
                    choice_rank=self.get_choice_rank,
                )

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                last_error = exc
                logger.warning(
                    "[SGLANG_CHAT_RETRY] attempt=%d/%d error=%s",
                    attempt,
                    self.max_retries,
                    exc,
                )

                if (
                    attempt < self.max_retries
                    and self.retry_delay_seconds > 0.0
                ):
                    time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            "SGLangChatAgent failed after "
            f"{self.max_retries} attempts."
        ) from last_error