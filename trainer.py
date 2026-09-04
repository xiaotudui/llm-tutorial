import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

class PretrainedDataset(Dataset):
    """读取 data_prepare_pretrained.py 生成的连续 token Bin 文件。"""

    def __init__(self, data_path: str | Path, max_length: int = 2048):
        if max_length < 2:
            raise ValueError("max_length 必须大于等于 2")

        data_path = Path(data_path)
        metadata_path = (
            data_path / "metadata.json" if data_path.is_dir()
            else data_path.parent / "metadata.json"
        )

        metadata = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

        if data_path.is_dir():
            bin_path = data_path / metadata.get("token_file", "train.bin")
        else:
            bin_path = data_path

        if not bin_path.is_file():
            raise FileNotFoundError(f"找不到 token 文件: {bin_path}")

        self.bin_path = bin_path
        self.max_length = max_length
        self.dtype = np.dtype(metadata.get("dtype", "uint32"))

        file_size = bin_path.stat().st_size
        if file_size % self.dtype.itemsize != 0:
            raise ValueError(
                f"{bin_path} 的大小不是 {self.dtype} 字节数的整数倍，文件可能不完整"
            )

        self.num_tokens = file_size // self.dtype.itemsize
        metadata_num_tokens = metadata.get("num_tokens")
        if metadata_num_tokens is not None and metadata_num_tokens != self.num_tokens:
            raise ValueError(
                "metadata.json 中的 num_tokens 与 Bin 文件大小不一致: "
                f"{metadata_num_tokens} != {self.num_tokens}"
            )

        # 延迟打开，避免 Windows 下 DataLoader 多进程复制 memmap 句柄的问题。
        self._tokens = None

    def __len__(self) -> int:
        return self.num_tokens // self.max_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"数据索引越界: {index}")

        if self._tokens is None:
            self._tokens = np.memmap(
                self.bin_path,
                dtype=self.dtype,
                mode="r",
            )

        start = index * self.max_length
        token_ids = np.asarray(
            self._tokens[start : start + self.max_length],
            dtype=np.int64,
        )
        input_ids = torch.from_numpy(token_ids)

        # CausalLM 会在模型内部将 labels 左移一位计算 next-token loss。
        return {
            "input_ids": input_ids,
            "labels": input_ids,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state


def parse_args():
    parser = argparse.ArgumentParser(description="使用 Bin token 数据预训练 Llama")
    parser.add_argument("--data-dir", default="data/pretrained_data")
    parser.add_argument("--output-dir", default="outputs/llama-pretrained")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        default=None,
        help="继续训练；不指定路径时自动使用 output-dir 中的最新 checkpoint",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_model(tokenizer, max_length: int) -> LlamaForCausalLM:
    """创建原 200M 架构；词表扩大后，当前 tokenizer 下约为 389M 参数。"""
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=max_length,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    return LlamaForCausalLM(config)


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    tokenizer_dir = data_dir / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(
            f"找不到本地 tokenizer: {tokenizer_dir}，请先运行 data_prepare_pretrained.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )
    train_dataset = PretrainedDataset(data_dir, max_length=args.max_length)
    if len(train_dataset) == 0:
        raise ValueError(
            f"token 数量 {train_dataset.num_tokens} 小于 max_length={args.max_length}"
        )

    model = build_model(tokenizer, args.max_length)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    use_tf32 = has_cuda and torch.cuda.get_device_capability(0)[0] >= 8
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"

    print(f"训练设备: {device_name}")
    print(f"词表大小: {len(tokenizer):,}")
    print(f"模型参数量: {num_parameters:,} ({num_parameters / 1e6:.2f}M)")
    print(
        f"训练样本: {len(train_dataset):,}，每条 {args.max_length:,} tokens，"
        f"共使用 {len(train_dataset) * args.max_length:,} tokens/epoch"
    )
    if not has_cuda:
        print("警告: 没有检测到 CUDA，389M 模型在 CPU 上训练会非常慢。")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        use_cache=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
