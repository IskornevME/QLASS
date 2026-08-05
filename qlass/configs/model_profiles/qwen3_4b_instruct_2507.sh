# Exact non-thinking Qwen checkpoint.

: "${POLICY_MODEL_ID:=Qwen/Qwen3-4B-Instruct-2507}"
: "${POLICY_MODEL_NAME:=Qwen3-4B-Instruct-2507}"

: "${POLICY_MODEL_PATH:=${MODEL_PATH%/}/${POLICY_MODEL_NAME}}"
: "${TOKENIZER_PATH:=${POLICY_MODEL_PATH}}"

: "${AGENT_CONFIG:=sglang_chat}"
: "${CRITIC_BACKEND:=llm_judge}"

# -----------------------------------------------------------------------------
# Optional original QLASS critic.
#
# These variables are used only when CRITIC_BACKEND=qnet.
# The actor still remains Qwen; the critic uses the original
# Llama SFT tokenizer/template and trained QNet checkpoint.
# -----------------------------------------------------------------------------

: "${QNET_BASE_MODEL_NAME:=Llama-2-7b-chat-hf}"

: "${QNET_SFT_MODEL_NAME:=${EXP_NAME}-${QNET_BASE_MODEL_NAME}-${TASK}-sft}"

# May be overridden with a nested inference checkpoint, for example:
# qlass-Llama-2-7b-chat-hf-alfworld-Q/infer-checkpoint-14382
: "${QNET_CHECKPOINT_NAME:=${EXP_NAME}-${QNET_BASE_MODEL_NAME}-${TASK}-Q}"

: "${QNET_PATH:=${MODEL_PATH%/}/${QNET_CHECKPOINT_NAME}}"

# Match the tokenizer used when the original QNet was trained.
: "${QNET_TOKENIZER_PATH:=${MODEL_PATH%/}/${QNET_SFT_MODEL_NAME}}"

# Used by qlass.data_utils.get_chat_template().
# This value must identify Llama-2 Chat rather than Qwen.
: "${QNET_MODEL_NAME:=${QNET_SFT_MODEL_NAME}}"

: "${QNET_MAX_PROMPT_TOKENS:=3800}"
: "${QNET_KEEP_FIRST_N:=3}"
: "${QNET_MIN_TAIL_MSGS:=4}"


# Deployment context cap. This is deliberately smaller than the native
# 262K context to reduce the SGLang KV-cache allocation.
: "${SGLANG_CONTEXT_LENGTH:=32768}"
: "${SERVER_TP:=1}"

# Policy generation.
: "${POLICY_MAX_PROMPT_TOKENS:=30000}"
: "${POLICY_MAX_NEW_TOKENS:=256}"
: "${POLICY_KEEP_FIRST_N:=3}"
: "${POLICY_MIN_TAIL_MSGS:=4}"

: "${POLICY_TEMPERATURE:=0.7}"
: "${POLICY_TOP_P:=0.8}"
: "${POLICY_TOP_K:=20}"
: "${POLICY_MIN_P:=0.0}"
: "${POLICY_PRESENCE_PENALTY:=0.0}"

# Independent LLM critic.
: "${CRITIC_MODEL_NAME:=${POLICY_MODEL_NAME}}"
: "${CRITIC_TOKENIZER_PATH:=${TOKENIZER_PATH}}"

: "${CRITIC_MAX_PROMPT_TOKENS:=16000}"
: "${CRITIC_MAX_NEW_TOKENS:=64}"
: "${CRITIC_KEEP_FIRST_N:=3}"
: "${CRITIC_MIN_TAIL_MSGS:=4}"

: "${CRITIC_TEMPERATURE:=0.0}"
: "${CRITIC_TOP_P:=1.0}"
: "${CRITIC_TOP_K:=-1}"

: "${CRITIC_FAILURE_MODE:=neutral}"
: "${CRITIC_NEUTRAL_SCORE:=0.5}"
: "${CRITIC_MAX_PARSE_ATTEMPTS:=3}"
: "${CRITIC_REQUEST_TIMEOUT:=300}"
: "${CRITIC_RETRY_DELAY_SECONDS:=1}"
