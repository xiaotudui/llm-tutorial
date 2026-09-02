from accelerate import init_empty_weights
from transformers import LlamaConfig, LlamaForCausalLM


config = LlamaConfig(
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=8192,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    tie_word_embeddings=False,
)

num_params = sum(parameter.numel() for parameter in model.parameters())
print(f"参数量：{num_params:,}")
print(f"参数量：{num_params / 1_000_000_000:.2f}B（{num_params / 100_000_000:.2f} 亿）")

print(f"FP32 理论内存占用：{num_params * 4 / 1_000_000_000:.2f} GB")
print(f"BF16/FP16 理论内存占用：{num_params * 2 / 1_000_000_000:.2f} GB")
