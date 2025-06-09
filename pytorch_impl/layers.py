import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Basic transformer block with self-attention and feed forward network."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.0, start_offset: int = 0, end_offset: int = 0):
        super().__init__()
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.GELU(),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None):
        if context is None:
            context = x
        q = x[:, self.start_offset: x.size(1) - self.end_offset, :]
        k = context
        v = context
        attn_out, _ = self.attn(q, k, v)
        x = self.norm1(q + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class CrossAttentionLayer(nn.Module):
    """Cross attention layer used in the model."""

    def __init__(self, local_dim: int, global_dim: int, n_heads: int = 8,
                 start_offset: int = 0, end_offset: int = 0, dropout: float = 0.0):
        super().__init__()
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.local_norm = nn.LayerNorm(local_dim)
        self.global_norm = nn.LayerNorm(global_dim)
        self.attn = nn.MultiheadAttention(local_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(local_dim)
        self.ff = nn.Sequential(
            nn.Linear(local_dim, local_dim // 2),
            nn.GELU(),
            nn.Linear(local_dim // 2, local_dim),
            nn.GELU(),
        )

    def forward(self, local_x, global_x):
        local_repr = self.local_norm(local_x)
        global_repr = self.global_norm(global_x)
        q = local_repr[:, self.start_offset: local_repr.size(1) - self.end_offset, :]
        k = global_repr
        v = global_repr
        attn_out, _ = self.attn(q, k, v)
        out = self.norm(q + attn_out)
        out = self.ff(out) + out
        return out


class CatEmbeddings(nn.Module):
    """Categorical embeddings with positional encodings."""

    def __init__(self, embed_dim: int, num_categories: int, seq_len: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_categories, embed_dim))
        self.positional = nn.Embedding(seq_len, embed_dim)
        self.register_buffer('positions', torch.arange(seq_len))

    def forward(self, x):
        # x expected shape: (batch, seq_len, num_categories)
        projected = torch.einsum('bsc,ce->bse', x, self.embedding)
        return projected + self.positional(self.positions)


class ConvBlock(nn.Module):
    """Convolutional block used inside DeepImpute."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv000 = nn.Conv1d(embed_dim, embed_dim, 3, padding='same')
        self.conv010 = nn.Conv1d(embed_dim, embed_dim, 5, padding='same')
        self.conv011 = nn.Conv1d(embed_dim, embed_dim, 7, padding='same')
        self.conv020 = nn.Conv1d(embed_dim, embed_dim, 7, padding='same')
        self.conv021 = nn.Conv1d(embed_dim, embed_dim, 15, padding='same')
        self.add = nn.Identity()
        self.conv100 = nn.Conv1d(embed_dim, embed_dim, 3, padding='same')
        self.bn0 = nn.BatchNorm1d(embed_dim)
        self.dw_conv = nn.Conv1d(embed_dim, embed_dim, 1, padding='same')
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # input x shape: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)
        xa = self.conv000(x)
        xb = self.conv010(xa)
        xb = self.conv011(xb)
        xc = self.conv020(xa)
        xc = self.conv021(xc)
        x_mix = xb + xc
        x_mix = self.conv100(x_mix)
        x_mix = self.bn0(x_mix)
        x_mix = self.dw_conv(x_mix)
        x_mix = self.bn1(x_mix)
        x_mix = self.act(x_mix)
        return x_mix.transpose(1, 2)
