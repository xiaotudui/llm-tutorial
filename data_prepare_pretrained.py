import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# 将 数据 转换为 token，并存储。用于后续的训练
# 200M 模型，按照 x 40，大约需要 8B token数据

def parse_args():
    parser = argparse.ArgumentParser(description="将文本语料编码为可供预训练使用的 token 文件")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="data/pretrained_data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tokens-limit", type=int, default=8_000_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "openbmb/Ultra-FineWeb-L3",
        "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
        split="train",
        streaming=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer 没有定义 eos_token_id")

    dtype = np.dtype("uint32")
    if len(tokenizer) > np.iinfo(dtype).max:
        raise ValueError(f"词表大小 {len(tokenizer)} 超出了 uint32 的范围")

    token_path = output_dir / "train.bin"
    num_tokens = 0

    with tqdm(
        total=args.tokens_limit,
        desc="Token processing",
        unit="token",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress, token_path.open("wb") as fout:
        for batch in ds.iter(batch_size=args.batch_size):
            if num_tokens >= args.tokens_limit:
                break
            texts = [text if isinstance(text, str) else "" for text in batch["content"]]
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]

            # EOS 既标记文档边界，也防止不同文档直接黏在一起。
            for token_ids in encoded:
                if not token_ids or token_ids[-1] != eos_id:
                    token_ids.append(eos_id)
                np.asarray(token_ids, dtype=dtype).tofile(fout)
                tokens_written = len(token_ids)
                num_tokens += tokens_written
                progress.update(min(tokens_written, args.tokens_limit - progress.n))
                if num_tokens >= args.tokens_limit:
                    break

    tokenizer.save_pretrained(output_dir / "tokenizer")
    metadata = {
        "tokenizer": args.tokenizer,
        "dtype": dtype.name,
        "eos_token_id": eos_id,
        "num_tokens": num_tokens,
        "token_file": token_path.name,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)

    size_gib = token_path.stat().st_size / 1024**3
    print(f"token 文件：{token_path} ({size_gib:.2f} GiB)")


if __name__ == "__main__":
    main()

