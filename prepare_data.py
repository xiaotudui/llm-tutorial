import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="将文本语料编码为可供预训练使用的 token 文件")
    parser.add_argument("--tokenizer", default="zai-org/GLM-5.2")
    parser.add_argument("--output-dir", default="data/ultrafineweb_zh_tokens")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-parts", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = [
        "data/ultrafineweb_zh_l3/multi_style/"
        f"part-{i:05d}-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet"
        for i in range(args.num_parts)
    ]
    ds = load_dataset(
        "openbmb/Ultra-FineWeb-L3",
        "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
        data_files=data_files,
        split="train",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer 没有定义 eos_token_id")

    # uint32 能容纳常见大词表的 token id；连续二进制文件可由 memmap 直接读取。
    dtype = np.dtype("uint32")
    if len(tokenizer) > np.iinfo(dtype).max:
        raise ValueError(f"词表大小 {len(tokenizer)} 超出了 uint32 的范围")

    token_path = output_dir / "train.bin"
    num_tokens = 0
    num_documents = 0

    with token_path.open("wb") as fout:
        for batch in ds.iter(batch_size=args.batch_size):
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
                num_tokens += len(token_ids)
                num_documents += 1

    tokenizer.save_pretrained(output_dir / "tokenizer")
    metadata = {
        "tokenizer": args.tokenizer,
        "dtype": dtype.name,
        "eos_token_id": eos_id,
        "num_tokens": num_tokens,
        "num_documents": num_documents,
        "token_file": token_path.name,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)

    size_gib = token_path.stat().st_size / 1024**3
    print(f"处理完成：{num_documents:,} 篇文档，{num_tokens:,} tokens")
    print(f"token 文件：{token_path} ({size_gib:.2f} GiB)")


if __name__ == "__main__":
    main()

