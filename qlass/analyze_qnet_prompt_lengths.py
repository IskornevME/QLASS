import argparse
import json
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.append("/home/m.iskornev/qlass/QLASS/")
from qlass.data_utils import (
    is_qwen3_model,
    render_chat_prompt,
)


def count_tokens(conversation, tokenizer, model_name):
    role_map = {
        "human": "user",
        "gpt": "assistant",
    }

    messages = [
        {
            "role": role_map[msg["from"]],
            "content": msg["value"],
        }
        for msg in conversation
    ]

    prompt = render_chat_prompt(
        messages=messages,
        tokenizer=tokenizer,
        model_path=model_name,
        add_generation_prompt=False,
    )

    return len(
        tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=not is_qwen3_model(
                model_name
            ),
        ).input_ids
    )


def main(args):
    with open(args.data_path, "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=False,
    )

    lengths = np.asarray(
        [
            count_tokens(
                item["conversations"],
                tokenizer,
                args.model_name,
            )
            for item in data
        ]
    )

    print("num examples:", len(lengths))

    for p in [50, 90, 95, 99]:
        print(
            f"p{p}:",
            np.percentile(lengths, p),
        )

    print("max:", lengths.max())

    print(
        f"fraction > {args.max_prompt_tokens}:",
        np.mean(lengths > args.max_prompt_tokens),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="/home/m.iskornev/qlass/QLASS/data/train/alfworld/qwen3_react_qnet/vanilla.jsonl")
    parser.add_argument("--tokenizer_path", default="/home/m.iskornev/qlass/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model_name", default="Qwen3-4B-Instruct-2507")

    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=8192,
    )

    main(parser.parse_args())