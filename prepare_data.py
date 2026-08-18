from datasets import load_dataset
from transformers import AutoTokenizer

data_files = [
    "data/ultrafineweb_zh_l3/multi_style/"
    f"part-{i:05d}-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet"
    for i in range(50)
]

ds = load_dataset(
    "openbmb/Ultra-FineWeb-L3",
    "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
    data_files=data_files,
    streaming=True
)

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
