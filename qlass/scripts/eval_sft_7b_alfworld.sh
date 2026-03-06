export CUDA_VISIBLE_DEVICES=0

model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

node_num=1  # number of GPUs
CUDA_VISIBLE_DEVICES=0

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
sg_worker_port="${SERVER_PORT:-21001}"
dist_port="${DIST_PORT:-10003}"

# Part 1: SFT stage
sft_data_path="data/train/${task}/${task}_sft.json"
batch_size=64
micro_batch_size=4
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${exp_name}-${model_name}-${task}-sft_run1
# sft_model_name=base-${model_name}
sg_worker_port=21001
mkdir -p logs outputs4

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --host "${SERVER_HOST}" --port ${sg_worker_port} --tp-size 1 >> logs/${exp_name}-sg_worker.log 2>&1 &

sg_worker_pid=$!
trap 'kill ${sg_worker_pid} 2>/dev/null || true' EXIT

python - <<'PY'
import socket, time, os
host=os.environ.get("SERVER_HOST","127.0.0.1")
port=int(os.environ.get("SERVER_PORT","21001"))
deadline=time.time()+180
while time.time()<deadline:
    try:
        s=socket.create_connection((host,port),timeout=1.0)
        s.close()
        print(f"SGLang is up on {host}:{port}")
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError(f"SGLang did not start in time on {host}:{port}")
PY

python3 -m qlass.inference --agent_config sglang_sft --model_name ${sft_model_name} --exp_config ${task} --split dev --num_icl_examples 1 --exp_name qlass_eval_sft --force_first