from datasets import load_dataset

ds = load_dataset(
    "openbmb/Ultra-FineWeb-L3",
    "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
    split="train",
    streaming=True
)


for i, sample in enumerate(ds):
    print(sample)

    if i >= 4:
        break