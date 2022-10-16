import torch


class AttentionHead(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._q = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features)
        self._k = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features)
        self._v = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features)
        self._norm = torch.sqrt(torch.as_tensor(self.out_features))

    def forward(self, x):
        q = self._q(x)
        k = self._k(x)
        v = self._v(x)
        m = torch.matmul(q, k.mT)
        m = m / self._norm
        m = torch.softmax(m, dim=-1)
        x = torch.matmul(m, v)
        return x


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, in_features, mid_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = self.in_features
        self.num_heads = num_heads

        self._heads = torch.nn.ModuleList([
            AttentionHead(in_features=self.in_features, out_features=self.mid_features)
            for _ in range(self.num_heads)
        ])
        self._overall_features = self.mid_features * self.num_heads
        self._proj = torch.nn.Linear(in_features=self._overall_features, out_features=self.out_features)

    def forward(self, x):
        x = torch.cat([h(x) for h in self._heads], dim=-1)
        x = self._proj(x)
        return x


class FFN(torch.nn.Module):

    def __init__(self, in_features, mid_features):
        super().__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = in_features
        self._mod = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=self.mid_features),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.mid_features, out_features=self.out_features),
        )

    def forward(self, x):
        x = self._mod(x)
        return x


class EncoderBlock(torch.nn.Module):

    def __init__(
        self,
        in_features,
        attn_features,
        attn_num_heads,
        ffn_features,
        dropout=0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.attn_features = attn_features
        self.attn_num_heads = attn_num_heads
        self.ffn_features = ffn_features
        self.out_features = self.in_features
        self.dropout = dropout

        self._attn = MultiHeadAttention(
            in_features=self.in_features,
            mid_features=self.attn_features,
            num_heads=self.attn_num_heads
        )
        self._attn_norm = torch.nn.LayerNorm(self.out_features)
        self._ffn = FFN(in_features=self.out_features, mid_features=self.ffn_features)
        self._ffn_norm = torch.nn.LayerNorm(self.out_features)
        self._drop = torch.nn.Dropout(self.dropout)

    def forward(self, x):
        attn = self._attn(x)
        attn = self._drop(attn)
        x = x + attn
        x = self._attn_norm(x)
        ffn = self._ffn(x)
        ffn = self._drop(ffn)
        x = x + ffn
        x = self._ffn_norm(x)
        return x


class Embeddings(torch.nn.Module):

    def __init__(self, size, num_pos, vocab_size, dropout=0.1):
        super().__init__()
        self.size = size
        self.num_pos = num_pos
        self.vocab_size = vocab_size
        self.dropout = dropout

        self._tokens = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.size,
            padding_idx=0
        )
        self._pos = torch.nn.Embedding(
            num_embeddings=self.num_pos,
            embedding_dim=self.size,
            padding_idx=0
        )
        self._drop = torch.nn.Dropout(self.dropout)

    def forward(self, ids, pos):
        ids = self._tokens(ids)
        pos = self._pos(pos)
        x = ids + pos
        x = self._drop(x)
        return x


class TransformerEncoder(torch.nn.Module):
    """Following original implementation from 'Attention Is All You Need'."""

    def __init__(
        self,
        vocab_size,
        num_pos,
        num_layers=8,
        dim=512,
        attn_features=64,
        attn_num_heads=8,
        ffn_features=2048,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_pos = num_pos
        self.num_layers = num_layers
        self.dim = dim
        self.attn_features = attn_features
        self.attn_num_heads = attn_num_heads
        self.ffn_features = ffn_features
        self.dropout = dropout

        self._emb = Embeddings(
            size=self.dim,
            num_pos=self.num_pos,
            vocab_size=self.vocab_size,
            dropout=self.dropout
        )
        self._blocks = torch.nn.ModuleList([
            EncoderBlock(
                in_features=self.dim,
                attn_features=self.attn_features,
                attn_num_heads=self.attn_num_heads,
                ffn_features=self.ffn_features,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

    def forward(self, ids, pos):
        x = self._emb(ids=ids, pos=pos)
        for block in self._blocks:
            x = block(x)
        return x
