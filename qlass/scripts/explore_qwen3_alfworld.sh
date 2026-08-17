MODEL_PROFILE="${MODEL_PROFILE:-qwen3_4b_instruct_2507}"
MODEL_PROFILE_PATH="/home/m.iskornev/qlass/QLASS/qlass/configs/model_profiles/${MODEL_PROFILE}.sh"

source "${MODEL_PROFILE_PATH}"

POLICY_MAX_NEW_TOKENS=2048
POLICY_TEMPERATURE=0.7

SGLANG_PYTHON="${SGLANG_PYTHON:-/home/m.iskornev/miniforge3/envs/qlass_qwen_server/bin/python}"
QLASS_PYTHON="${QLASS_PYTHON:-/home/m.iskornev/miniforge3/envs/my_env/bin/python}"


task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

num_workers=8   # number of inference workers
num_server=2

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/train/${task}/${task}_sft.json"
batch_size=64
micro_batch_size=4

sg_worker_port=21001
sg_worker_port3=21003

CUDA_VISIBLE_DEVICES=0 "${SGLANG_PYTHON}" -m sglang.launch_server --model-path "${POLICY_MODEL_PATH}" --context-length "${SGLANG_CONTEXT_LENGTH}" --port ${sg_worker_port} >> logs/${exp_name}-sg_worker_explore1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 "${SGLANG_PYTHON}" -m sglang.launch_server --model-path "${POLICY_MODEL_PATH}" --context-length "${SGLANG_CONTEXT_LENGTH}" --port ${sg_worker_port3} >> logs/${exp_name}-sg_worker_explore3.log 2>&1 &

sleep 60

for ((i=0;i<${num_workers};i=i+1)); do
    echo "Start worker ${i} querying server $((i%$num_server))"
    CUDA_VISIBLE_DEVICES="" "${QLASS_PYTHON}" qlass/explore_sft_agent.py \
        --agent_config sglang_chat \
        --policy_server_address "http://127.0.0.1:${current_port}" \
        --model_name "${POLICY_MODEL_NAME}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --policy_max_prompt_tokens "${POLICY_MAX_PROMPT_TOKENS}" \
        --policy_max_new_tokens 2048 \
        --policy_temperature 0.7 \
        --policy_top_p 0.8 \
        --policy_top_k 20 \
        --policy_min_p 0.0 \
        --policy_presence_penalty 0.0 \
        --alfworld_react_prompt \
        --alfworld_history_length 50 \
        --max_steps 50 \
        --agent_path qlass/configs/model/ \
        --exp_name ${exp_name} \
        --exp_path qlass/configs/task/ \
        --exp_config ${task} \
        --split train \
        --slice_num ${num_workers} \
        --slice_id ${i} \
        --model_name ${sft_model_name} \
        --max_depth 8 \
        --min_prune_depth 3 \
        --num_icl_examples 0 \
        --samples_per_depth 2 \
        --output_dir data/train/${task}/explore_qwen3_react_d8_s2_mpr3/ &
done
