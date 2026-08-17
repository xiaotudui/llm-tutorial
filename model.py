from dataclasses import dataclass
from typing import Optional

from torch import nn
from torch.nn import functional as F
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)




@dataclass
class TuduiGPTConfig:
    vocab_size: int = 32000
    dim: int = 384
    n_layers: int = 16
    n_heads: int = 6
    n_kv_heads: int = 6
    hidden_dim: Optional[int] = None
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.hidden_dim is None:
            # Llama-style SwiGLU width, rounded for friendlier matrix sizes.
            width = int(8 * self.dim / 3)
            self.hidden_dim = 256 * ((width + 255) // 256)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TuduiGPTConfig, rope: LlamaRotaryEmbedding):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.dropout = config.dropout
        self.rope = rope

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        past_len = 0 if past_kv is None else past_kv[0].shape[-2]

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Hugging Face's Llama RoPE is applied only to new Q/K. Cached K is
        # already rotated, so its positions are represented by ``past_len``.
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=x.device
        ).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.rope(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present_kv = (k, v) if use_cache else None

        # GQA: expand K/V heads only for attention; keep the compact form in cache.
        if self.n_rep > 1:
            k_for_attn = k.repeat_interleave(self.n_rep, dim=1)
            v_for_attn = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_for_attn, v_for_attn = k, v

        attn_mask = None
        is_causal = past_len == 0
        if past_len > 0 and seq_len > 1:
            # Query i may see all cached tokens and current tokens through i.
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            key_positions = torch.arange(past_len + seq_len, device=x.device)
            attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        output = F.scaled_dot_product_attention(
            q,
            k_for_attn,
            v_for_attn,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.resid_dropout(self.out_proj(output)), present_kv


class TransformerBlock(nn.Module):
    def __init__(self, config: TuduiGPTConfig, rope: LlamaRotaryEmbedding):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config, rope)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = SwiGLU(config.dim, config.hidden_dim, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_output, present_kv = self.attention(
            self.attn_norm(x), past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_output
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, present_kv


class TuduiGPT(nn.Module):
    """Decoder-only language model following the Mini-LLM/Llama-style layout."""

    def __init__(self, config: Optional[TuduiGPTConfig] = None, **config_kwargs):
        super().__init__()
        if config is not None and config_kwargs:
            raise ValueError("Pass either config or keyword config values, not both")
        self.config = config or TuduiGPTConfig(**config_kwargs)

        self.token_embedding = TokenEmbedding(self.config.vocab_size, self.config.dim)
        self.embedding_dropout = nn.Dropout(self.config.dropout)

        # Use Transformers' maintained Llama RoPE implementation. One shared
        # module is reused by every attention layer.
        rope_config = LlamaConfig(
            hidden_size=self.config.dim,
            num_hidden_layers=self.config.n_layers,
            num_attention_heads=self.config.n_heads,
            num_key_value_heads=self.config.n_kv_heads,
            intermediate_size=self.config.hidden_dim,
            max_position_embeddings=self.config.max_seq_len,
            rope_theta=self.config.rope_base,
        )
        self.rope = LlamaRotaryEmbedding(rope_config)
        self.layers = nn.ModuleList(
            TransformerBlock(self.config, self.rope) for _ in range(self.config.n_layers)
        )
        self.norm = nn.RMSNorm(self.config.dim, eps=self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.token_embedding.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict[str, object]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one (K, V) pair per layer")

        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if past_len + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input and KV cache exceed max_seq_len")

        x = self.embedding_dropout(self.token_embedding(input_ids))
        new_key_values = [] if use_cache else None

        for layer_index, layer in enumerate(self.layers):
            past_kv = None if past_key_values is None else past_key_values[layer_index]
            x, present_kv = layer(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_key_values.append(present_kv)

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if input_ids.shape[1] < 2:
                raise ValueError("At least two tokens are required to compute next-token loss")
            # Token at position t predicts the token at position t + 1.
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                targets[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_key_values,
        }


if __name__ == "__main__":
    config = TuduiGPTConfig(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
    )
    model = TuduiGPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    output = model(tokens, targets=tokens)
    print("logits:", output["logits"].shape)
    print("loss:", output["loss"].item())
