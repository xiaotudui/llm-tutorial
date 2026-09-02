"""基于 Hugging Face Llama 的精简预训练模型。"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def build_model(
    vocab_size: int = 32_000,
    dim: int = 384,
    n_layers: int = 16,
    n_heads: int = 6,
    n_kv_heads: int = 6,
    max_seq_len: int = 2_048,
    dropout: float = 0.0,
) -> LlamaForCausalLM:
    """创建一个随机初始化、可直接进行因果语言模型预训练的 Mini-LLM。"""
    if dim % n_heads != 0:
        raise ValueError("dim 必须能被 n_heads 整除")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads 必须能被 n_kv_heads 整除")

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=dim,
        intermediate_size=4 * dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=max_seq_len,
        attention_dropout=dropout,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config)


if __name__ == "__main__":
    model = build_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=input_ids)
    print(f"parameters: {model.num_parameters():,}")
    print(f"logits: {tuple(output.logits.shape)}, loss: {output.loss.item():.4f}")
