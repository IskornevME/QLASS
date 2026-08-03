# General non-benchmark-specific Qwen model.

: "${POLICY_MODEL_NAME:=Qwen3-4B-Instruct-2507}"

: "${POLICY_MODEL_PATH:=${MODEL_PATH%/}/${POLICY_MODEL_NAME}}"
: "${TOKENIZER_PATH:=${POLICY_MODEL_PATH}}"

: "${AGENT_CONFIG:=sglang_chat}"
: "${CRITIC_BACKEND:=llm_judge}"

# It is substantially larger than Llama-2's 4K,
# but considerably cheaper than allocating the full 262K KV cache.
: "${SGLANG_CONTEXT_LENGTH:=32768}"

: "${POLICY_MAX_PROMPT_TOKENS:=30000}"
: "${POLICY_MAX_NEW_TOKENS:=256}"
: "${POLICY_KEEP_FIRST_N:=3}"
: "${POLICY_MIN_TAIL_MSGS:=4}"

# Recommended Qwen3-Instruct-2507 sampling values.
: "${POLICY_TEMPERATURE:=0.7}"
: "${POLICY_TOP_P:=0.8}"
: "${POLICY_TOP_K:=20}"
: "${POLICY_MIN_P:=0.0}"
: "${POLICY_PRESENCE_PENALTY:=0.0}"

# LLM judge uses the same served model.
: "${CRITIC_MODEL_PATH:=${POLICY_MODEL_PATH}}"
: "${CRITIC_MAX_PROMPT_TOKENS:=16000}"
: "${CRITIC_MAX_NEW_TOKENS:=256}"
: "${CRITIC_TEMPERATURE:=0.0}"
: "${CRITIC_TOP_P:=1.0}"
: "${CRITIC_TOP_K:=-1}"
: "${CRITIC_RECENT_ACTIONS:=8}"
: "${CRITIC_FAILURE_MODE:=neutral}"

: "${SERVER_TP:=1}"