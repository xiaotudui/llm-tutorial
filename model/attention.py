from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, n_heads, n_kv_heads, dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dim = dim
        self.head_dim = self.dim // self.n_heads
        self.wq = nn.Linear(self.dim, self.head_dim * self.n_heads)
        self.wk = nn.Linear(self.dim, self.head_dim * self.n_kv_heads)
        self.wv = nn.Linear(self.dim, self.head_dim * self.n_kv_heads)
        self.repeat_kv = self.n_heads // self.n_kv_heads
        

    def forward(self, x, mask):
        x_q, x_k, x_v = self.wq(x), self.wk(x), self.wv(x)
        batch_size, seq_len, token_dim = x.shape
        x_q = x_q.view(batch_size, seq_len, self.n_head, self.head_dim)
        x_k = x_k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        x_v = x_v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if self.repeat_kv > 1:
            x_k = x_k.repeat_interleave(self.repeat_kv, dim = 2)
            x_v = x_v.repeat_interleave(self.repeat_kv, dim = 2)

        x_q = x_q.transpose(1, 2)
        x_k = x_k.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        output = F.scaled_dot_product_attention(x_q, x_k, x_v, mask, dropout_p=0.0, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return x