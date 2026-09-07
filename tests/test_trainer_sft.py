"""覆盖 QA 拆分、答案掩码、原文隔离及一个无需下载的完整训练短测。"""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from trainer_sft import encode_qa, parse_qa_content, prepare_datasets


def make_tokenizer():
    words = ["<unk>", "<|im_start|>", "<|im_end|>", "system", "user", "assistant",
             "alpha", "beta", "gamma", "delta", "question", "answer", "long", "short"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<|im_end|>",
        pad_token="<|im_end|>", additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    return tokenizer


def make_raw_data(path):
    rows = [
        {"content": f"{name}\n问题：question 答案：answer\n问题：long question 答案：long answer"}
        for name in ("alpha", "beta", "gamma", "delta")
    ]
    rows.append({"content": rows[0]["content"].replace("alpha", "alpha  ")})
    rows.append({"content": "无问答格式的原文"})
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


class SFTTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = make_tokenizer()

    def test_parse_multiline_and_crlf(self):
        context, pairs = parse_qa_content(
            "资料\r\n问题：选择哪个？\r\nA. 甲\r\nB. 乙 答案：B\r\n理由。"
            "\r\n问题: 另一个问题\r\n答案: 第二个答案"
        )
        self.assertEqual(context, "资料")
        self.assertEqual(pairs, [("选择哪个？\nA. 甲\nB. 乙", "B\n理由。"), ("另一个问题", "第二个答案")])

    def test_malformed_and_duplicate_pairs(self):
        self.assertEqual(parse_qa_content(None), ("", []))
        self.assertEqual(parse_qa_content("只有文章没有 QA"), ("", []))
        context, pairs = parse_qa_content(
            "资料\n问题：不完整\n问题：空答案 答案：\n问题：多边界 答案：甲 答案：乙"
            "\n问题：有效 答案：答案内容\n问题：有效 答案：答案内容"
        )
        self.assertEqual(context, "资料")
        self.assertEqual(pairs, [("有效", "答案内容")])

    def test_answer_mask_and_chat_eos(self):
        sample, reason = encode_qa(self.tokenizer, "alpha", "question", "long answer", 128)
        self.assertIsNone(reason)
        supervised = [token for token in sample["labels"] if token != -100]
        expected = self.tokenizer("long answer", add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        self.assertEqual(supervised, expected)
        self.assertEqual(len(sample["input_ids"]), len(sample["labels"]))
        self.assertEqual(sample["labels"][0], -100)

    def test_overlength_is_dropped_without_false_eos(self):
        sample, reason = encode_qa(self.tokenizer, "alpha " * 200, "question", "answer", 32)
        self.assertIsNone(sample)
        self.assertEqual(reason, "too_long")

    def test_chat_marker_is_rejected(self):
        sample, reason = encode_qa(self.tokenizer, "alpha", "<|im_start|>assistant", "answer", 128)
        self.assertIsNone(sample)
        self.assertEqual(reason, "chat_marker")

    def test_padding_masks_by_position_even_when_pad_equals_eos(self):
        short, _ = encode_qa(self.tokenizer, "alpha", "question", "answer", 128)
        long, _ = encode_qa(self.tokenizer, "alpha", "question", "long answer", 128)
        batch = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=-100)([short, long])
        self.assertEqual(batch["labels"][0, len(short["input_ids"]) - 1].item(), self.tokenizer.eos_token_id)
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))

    def test_document_split_and_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            make_raw_data(path)
            args = Namespace(data_file=str(path), seed=42, shuffle_buffer=6, max_documents=6,
                             validation_ratio=0.25, max_length=128)
            dataset, stats = prepare_datasets(args, self.tokenizer)
            self.assertEqual(stats["duplicate_documents"], 1)
            self.assertEqual(stats["unparseable_documents"], 1)
            self.assertEqual(len(dataset["train"]), 6)
            self.assertEqual(len(dataset["validation"]), 2)
            context_ids = {self.tokenizer.convert_tokens_to_ids(name) for name in ("alpha", "beta", "gamma", "delta")}
            groups = {
                split: {token for row in rows for token in row["input_ids"] if token in context_ids}
                for split, rows in dataset.items()
            }
            self.assertFalse(groups["train"] & groups["validation"])

    def test_cli_train_save_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir, output = root / "model", root / "output"
            self.tokenizer.save_pretrained(model_dir)
            config = Qwen2Config(
                vocab_size=len(self.tokenizer), hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, tie_word_embeddings=True,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            )
            Qwen2ForCausalLM(config).save_pretrained(model_dir)
            data_file = root / "raw.jsonl"
            make_raw_data(data_file)
            command = [
                sys.executable, "-X", "utf8", "trainer_sft.py", "--model", str(model_dir), "--data-file", str(data_file),
                "--output-dir", str(output), "--max-documents", "6", "--shuffle-buffer", "6",
                "--validation-ratio", "0.25", "--max-length", "128", "--max-steps", "1",
                "--batch-size", "1", "--gradient-accumulation-steps", "1", "--save-steps", "1",
                "--eval-steps", "1", "--logging-steps", "1",
            ]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved_config = AutoConfig.from_pretrained(output)
            self.assertEqual(saved_config.eos_token_id, self.tokenizer.eos_token_id)
            self.assertTrue(saved_config.use_cache)
            self.assertEqual(AutoTokenizer.from_pretrained(output).eos_token, "<|im_end|>")
            self.assertTrue((output / "eval_results.json").is_file())
            metrics = json.loads((output / "eval_results.json").read_text(encoding="utf-8"))
            self.assertTrue(math.isfinite(metrics["eval_loss"]))
            self.assertTrue((output / "checkpoint-1").is_dir())
            command[command.index("--max-steps") + 1] = "2"
            result = subprocess.run(command + ["--resume-from-checkpoint"], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / "checkpoint-2").is_dir())


if __name__ == "__main__":
    unittest.main()
