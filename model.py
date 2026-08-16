from torch import nn
import torch

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, token_dim):
        super().__init__()
        # todo: padding idx
        self.token_embedding = nn.Embedding(vocab_size, token_dim)

    def forward(self, x):
        return self.token_embedding(x)


class RoPE(nn.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.angle_rates = 1 / torch.pow(10000, torch.arange(0, dim, 2).float() / dim)


    def forward(self, x):
        return x


class TuduiGPT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x


if __name__ == "__main__":
    print("hello")

