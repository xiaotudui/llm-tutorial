from transformers import LlamaConfig, LlamaForCausalLM

# 8B Model 
# config = LlamaConfig(
#     vocab_size=128256,
#     hidden_size=4096,
#     intermediate_size=14336,
#     num_hidden_layers=32,
#     num_attention_heads=32,
#     num_key_value_heads=8,
#     max_position_embeddings=8192,
#     rms_norm_eps=1e-5,
#     rope_theta=500000.0,
#     tie_word_embeddings=False,
# )

# 1.1B
# config = LlamaConfig(
#     vocab_size=32000,
#     hidden_size=2048,
#     intermediate_size=5632,
#     num_hidden_layers=22,
#     num_attention_heads=32,
#     num_key_value_heads=4,
#     max_position_embeddings=2048,
#     rms_norm_eps=1e-5,
#     rope_theta=10000.0,
#     tie_word_embeddings=False,
# )

# 200M
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=24,
    num_attention_heads=12,
    num_key_value_heads=4,
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=False,
)

model = LlamaForCausalLM(config)

num_params = sum(parameter.numel() for parameter in model.parameters())
print(f"参数量：{num_params:,}")
print(f"参数量：{num_params / 1_000_000_000:.2f}B（{num_params / 100_000_000:.2f} 亿）")
