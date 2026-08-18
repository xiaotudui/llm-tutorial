import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from transformers import AutoTokenizer

data_files = [
    "data/ultrafineweb_zh_l3/multi_style/"
    f"part-{i:05d}-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet"
    for i in range(1)
]

ds = load_dataset(
    "openbmb/Ultra-FineWeb-L3",
    "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
    data_files=data_files
)

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
eos_id = tokenizer.eos_token_id
if eos_id is None:
    raise ValueError("Tokenizer 没有定义 eos_token_id")

all_tokens = []
for item in ds["train"]:
    tokens = tokenizer.encode(item["content"], add_special_tokens=False)
    if not tokens or tokens[-1] != eos_id:
        tokens.append(eos_id)
    all_tokens.extend(tokens)

