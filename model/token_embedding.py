from torch import nn
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, token_dim):
        super().__init__()
        # todo: padding idx
        self.token_embedding = nn.Embedding(vocab_size, token_dim)

    def forward(self, x):
        return self.token_embedding(x)
    