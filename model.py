from torch import nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self, n_heads, n_kv_heads, dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dim = dim
        self.head_dim = self.dim // self.n_heads
        self.wq = nn.Linear(self.dim, self.head_dim * self.n_heads, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(self.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.repeat_kv = self.n_heads // self.n_kv_heads

    def forward(self, x):
        x_q, x_k, x_v = self.wq(x), self.wk(x), self.wv(x)
        batch_size, seq_len, token_dim = x.shape
        x_q = x_q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        x_k = x_k.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        x_v = x_v.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # todo: RoPE to q, k
        if self.repeat_kv > 1:
            x_k = x_k.repeat_interleave(self.repeat_kv, dim=2)
            x_v = x_v.repeat_interleave(self.repeat_kv, dim=2)

        x_q = x_q.transpose(1, 2)
        x_k = x_k.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            x_q, x_k, x_v, dropout_p=0.0, is_causal=True
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self):
        
        

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, n_kv_heads, dim):
        super().__init__()
        self.attention = Attention(n_heads=n_heads, n_kv_heads=n_kv_heads, dim=dim)
        self.attention_norm = nn.RMSNorm(dim)

    def forward(self, x):
        attention_result = self.attention(self.attention_norm(x))
        h = x + attention_result
        return h


class TuduiGPT(nn.Module):
    def __init__(self, vocab_size, dim, n_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(TransformerBlock(6, 6, dim))

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    vocab_size = 36000
    dim = 512
    n_layers = 2
    n_heads = 6
    n_kv_heads = 6

    model = TuduiGPT(vocab_size=36000, dim=512, n_layers=2)
    input = torch.randint(1, 10, (5, 3))
    print(input.shape)
    output = model(input)
    print(output.shape)
