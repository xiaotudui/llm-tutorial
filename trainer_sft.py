"""将 Ultra-FineWeb-L3 中文 QA 转为阅读理解 SFT，微调 Qwen2.5-0.5B。"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


DATASET_NAME = "openbmb/Ultra-FineWeb-L3"
DATASET_CONFIG = "Ultra-FineWeb-L3-zh-QA-Synthetic"
SYSTEM_PROMPT = "你是一个中文助手，请根据用户提供的参考资料准确回答问题。"
# 实际数据格式：原文 + 多个以行首“问题：”开始的问答，答案可能在同一行。
QUESTION_PATTERN = re.compile(r"(?m)^[ \t]*问题[ \t]*[：:][ \t]*")
ANSWER_PATTERN = re.compile(r"答案[ \t]*[：:][ \t]*")
CHAT_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def parse_qa_content(content: str) -> tuple[str, list[tuple[str, str]]]:
    """只接受明确的问题/答案边界；不把未解析的整篇文章当作助手答案。"""
    if not isinstance(content, str) or not content.strip():
        return "", []
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    questions = list(QUESTION_PATTERN.finditer(content))
    if not questions:
        return "", []
    context = content[: questions[0].start()].strip()
    pairs = []
    seen = set()
    for index, question in enumerate(questions):
        end = questions[index + 1].start() if index + 1 < len(questions) else len(content)
        block = content[question.end() : end]
        # 多个“答案：”意味着边界不确定，跳过以免错误监督。
        answers = list(ANSWER_PATTERN.finditer(block))
        if len(answers) != 1:
            continue
        answer_marker = answers[0]
        pair = (block[: answer_marker.start()].strip(), block[answer_marker.end() :].strip())
        if all(pair) and pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    return context, pairs


def encode_qa(tokenizer, context: str, question: str, answer: str, max_length: int):
    """使用模型自带 chat template；仅答案和结束标记参与 next-token loss。"""
    if any(marker in text for marker in CHAT_MARKERS for text in (context, question, answer)):
        return None, "chat_marker"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
    )["input_ids"]
    input_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template 的助手前缀不一致，无法可靠构建答案 loss mask")
    if len(input_ids) > max_length:
        # 原文缺失会导致问题无法回答，截断答案会错误教会提前结束，因此整条跳过。
        return None, "too_long"
    if len(input_ids) <= len(prompt_ids):
        return None, "empty_answer"
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}, None


def prepare_datasets(args, tokenizer) -> tuple[DatasetDict, dict]:
    if args.data_file:
        source = load_dataset("json", data_files=args.data_file, split="train", streaming=True)
    else:
        source = load_dataset(
            DATASET_NAME, DATASET_CONFIG, split="train", streaming=True,
            revision=args.dataset_revision,
        )
    source = source.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    documents = []
    seen_contexts = set()
    stats = Counter()
    for row in tqdm(source.take(args.max_documents), total=args.max_documents, desc="读取 QA 原文"):
        stats["source_documents"] += 1
        if "content" not in row:
            raise ValueError("数据必须包含 content 字段，格式为原文 + 问题：…答案：…")
        context, pairs = parse_qa_content(row["content"])
        if not context or not pairs:
            stats["unparseable_documents"] += 1
            continue
        # 同一原文的所有 QA 必须留在同一个 split；精确去重忽略空白差异。
        source_id = hashlib.sha256("".join(context.split()).encode("utf-8")).hexdigest()
        if source_id in seen_contexts:
            stats["duplicate_documents"] += 1
            continue
        seen_contexts.add(source_id)
        documents.append({
            "source_id": source_id,
            "context": context,
            "questions": [q for q, _ in pairs],
            "answers": [a for _, a in pairs],
        })
        stats["parsed_pairs"] += len(pairs)
    if not documents:
        raise ValueError("未解析出有效的带原文 QA，请检查数据格式或增加 --max-documents")
    raw = Dataset.from_list(documents)
    if args.validation_ratio:
        if len(raw) < 2:
            raise ValueError("按原文划分验证集至少需要两篇有效原文；短测可设 --validation-ratio 0")
        split = raw.train_test_split(test_size=args.validation_ratio, seed=args.seed)
        raw_splits = DatasetDict(train=split["train"], validation=split["test"])
    else:
        raw_splits = DatasetDict(train=raw)

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")),
    })
    encoded = DatasetDict()
    for name, raw_split in raw_splits.items():
        stats[f"{name}_documents"] = len(raw_split)

        def encode_batch(batch):
            output = {key: [] for key in features}
            for context, questions, answers in zip(batch["context"], batch["questions"], batch["answers"]):
                for question, answer in zip(questions, answers):
                    sample, reason = encode_qa(tokenizer, context, question, answer, args.max_length)
                    if sample is None:
                        stats[f"{name}_skipped_{reason}"] += 1
                        continue
                    for key in output:
                        output[key].append(sample[key])
            return output

        encoded[name] = raw_split.map(
            encode_batch, batched=True, batch_size=16, remove_columns=raw_split.column_names,
            features=features, load_from_cache_file=False, desc=f"编码 {name}（仅监督答案）",
        )
        stats[f"{name}_samples"] = len(encoded[name])
        if not len(encoded[name]):
            raise ValueError(f"{name} 无有效样本；请增大 --max-length / --max-documents 并检查原文格式")
    return encoded, dict(stats)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--data-file", help="可选：本地 JSONL，包含与远程数据相同的 content 字段")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-0.5b-sft")
    parser.add_argument("--max-documents", type=int, default=10000, help="最多读取的原始文档数，一篇可拆出多组 QA")
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest", default=None)
    parser.add_argument("--prepare-only", action="store_true", help="保存编码后的数据和预览，不下载模型权重或训练")
    args = parser.parse_args()
    for name in ("max_documents", "shuffle_buffer", "batch_size", "gradient_accumulation_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.max_length < 32:
        parser.error("--max-length 必须至少为 32")
    if not 0 <= args.validation_ratio < 1:
        parser.error("--validation-ratio 必须在 [0, 1) 内")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio 必须在 [0, 1) 内")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if not tokenizer.chat_template:
        raise ValueError("模型 tokenizer 必须提供 chat_template")
    chat_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if chat_end_id is None or chat_end_id == tokenizer.unk_token_id:
        raise ValueError("本脚本要求 Qwen ChatML tokenizer，缺少 <|im_end|>")
    # Qwen Base 默认 eos 是 endoftext；SFT 后应在助手的 im_end 处停止生成。
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    datasets, stats = prepare_datasets(args, tokenizer)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    sample = datasets["train"][0]
    preview = {
        "conversation": tokenizer.decode(sample["input_ids"], skip_special_tokens=False),
        "supervised_answer": tokenizer.decode([t for t in sample["labels"] if t != -100], skip_special_tokens=False),
    }
    (output_dir / "sft_preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "sft_data_stats.json").write_text(
        json.dumps({"arguments": vars(args), "statistics": stats}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if args.prepare_only:
        datasets.save_to_disk(str(output_dir / "prepared_dataset"))
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print(f"数据已保存到 {output_dir}，尚未加载模型权重或训练。")
        return

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, attn_implementation="sdpa", dtype=torch.float32,
    )
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("--max-length 超出模型支持的上下文长度")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = chat_end_id
    model.generation_config.eos_token_id = chat_end_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"全参数 SFT: {sum(p.numel() for p in model.parameters()):,} 参数")
    print(f"训练设备: {torch.cuda.get_device_name(0) if has_cuda else 'CPU'}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        # 本项目 transformers 5.15 支持 [0, 1) 的 warmup_steps 表示比例。
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=use_bf16,
        fp16=has_cuda and not use_bf16,
        tf32=has_cuda and torch.cuda.get_device_capability(0)[0] >= 8,
        gradient_checkpointing=args.gradient_checkpointing,
        use_cache=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if "validation" in datasets else "no",
        eval_steps=args.eval_steps,
        prediction_loss_only=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=has_cuda,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    resume = True if args.resume_from_checkpoint == "latest" else args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)
    if "validation" in datasets:
        trainer.save_metrics("eval", trainer.evaluate())
    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
