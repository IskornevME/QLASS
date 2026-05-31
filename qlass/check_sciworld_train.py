import sys
import json
import statistics
import transformers
import os
root_dir = os.getcwd()
from pathlib import Path
sys.path.append(root_dir)
sys.path.append((Path(root_dir).parent))
from qlass.data_utils import get_chat_template

MODEL_PATH = "/home/m.iskornev/qlass/models/base-Llama-2-7b-chat-hf"
DATA_PATH = "/home/m.iskornev/qlass/QLASS/data/train/sciworld/sciworld_sft.json"
MAX_LEN = 4096

tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_PATH,
    model_max_length=MAX_LEN,
    padding_side="right",
    use_fast=False,
)
if tokenizer.pad_token != tokenizer.unk_token:
    tokenizer.pad_token = tokenizer.unk_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.unk_token = tokenizer.pad_token

chat = get_chat_template(MODEL_PATH)
roles = {"human": "user", "gpt": "assistant"}
roles_list = ["user", "assistant"]

data = json.load(open(DATA_PATH))
lengths = []

for ex in data:
    source = ex["conversations"]
    if roles[source[0]["from"]] != "user":
        source = source[1:]

    messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == roles_list[j % 2], ex["id"]
        messages.append({"role": role, "content": sentence["value"]})

    prompt = chat.get_prompt(messages)
    n_tokens = len(tokenizer(prompt, add_special_tokens=True).input_ids)
    lengths.append((ex["id"], n_tokens))

only_lengths = [x[1] for x in lengths]
over = [x for x in lengths if x[1] > MAX_LEN]

print("num examples:", len(lengths))
print("mean:", statistics.mean(only_lengths))
print("median:", statistics.median(only_lengths))
print("p95:", sorted(only_lengths)[int(0.95 * len(only_lengths))])
print("max:", max(only_lengths))
print("num > 4096:", len(over))
print("share > 4096:", len(over) / len(lengths))
print("top 20 longest:")
for x in sorted(lengths, key=lambda z: z[1], reverse=True)[:20]:
    print(x)