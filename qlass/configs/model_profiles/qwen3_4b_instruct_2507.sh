# Exact non-thinking Qwen checkpoint.

: "${POLICY_MODEL_ID:=Qwen/Qwen3-4B-Instruct-2507}"
: "${POLICY_MODEL_NAME:=Qwen3-4B-Instruct-2507}"

: "${POLICY_MODEL_PATH:=${MODEL_PATH%/}/${POLICY_MODEL_NAME}}"
: "${TOKENIZER_PATH:=${POLICY_MODEL_PATH}}"

: "${AGENT_CONFIG:=sglang_chat}"
: "${CRITIC_BACKEND:=llm_judge}"

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
