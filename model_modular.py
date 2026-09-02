import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class TuduiModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        return self.norm(hidden_states)


class TuduiForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 36_000,
        hidden_size: int = 384,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 6,
        max_position_embeddings: int = 2_048,
    ):
        super().__init__()

        self.config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size or 4 * hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            use_cache=False,
        )
        self.config._attn_implementation = "eager"

        self.model = TuduiModel(self.config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.apply(self._init_weights)

        self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutput(loss=loss, logits=logits)


if __name__ == "__main__":
    model = TuduiForCausalLM(
        vocab_size=1_000,
        hidden_size=96,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    outputs = model(input_ids=input_ids, labels=input_ids)

    print(model)
    print(f"logits: {tuple(outputs.logits.shape)}")
    print(f"loss: {outputs.loss.item():.4f}")
