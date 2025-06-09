import torch
from torch import nn
from .layers import TransformerBlock, CrossAttentionLayer, CatEmbeddings, ConvBlock


class ChunkModule(nn.Module):
    """A single chunk processing module used inside DeepImpute."""

    def __init__(self, input_len: int, embed_dim: int, num_heads: int,
                 start_offset: int = 0, end_offset: int = 0, dropout_rate: float = 0.25):
        super().__init__()
        self.transformer = TransformerBlock(embed_dim, num_heads, embed_dim // 2,
                                            dropout=0.0, start_offset=start_offset,
                                            end_offset=end_offset)
        self.conv = ConvBlock(embed_dim)
        self.conv_skip = ConvBlock(embed_dim)
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.conv_after = ConvBlock(embed_dim)
        self.cross = CrossAttentionLayer(embed_dim, embed_dim, start_offset=0, end_offset=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv_final = ConvBlock(embed_dim)
        self.concat = lambda x, y: torch.cat([x, y], dim=-1)

    def forward(self, x):
        xa0 = self.transformer(x, x)
        xa = self.conv(xa0)
        xa_skip = self.conv_skip(xa)
        xa = self.dense(xa)
        xa = self.conv_after(xa)
        xa = self.cross(xa, xa0)
        xa = self.dropout(xa)
        xa = self.conv_final(xa)
        xa = self.concat(xa_skip, xa)
        return xa


class DeepImpute(nn.Module):
    """Simplified PyTorch implementation of the DeepImpute model."""

    def __init__(self, seq_len: int, in_channel: int, embed_dim: int, num_heads: int,
                 chunk_size: int = 2048, attention_range: int = 64,
                 offset_before: int = 0, offset_after: int = 0, dropout_rate: float = 0.25):
        super().__init__()
        self.seq_len = seq_len
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.attention_range = attention_range
        self.offset_before = offset_before
        self.offset_after = offset_after

        self.embedding = CatEmbeddings(embed_dim, in_channel, seq_len)
        self._build_chunks()
        self.concat = lambda tensors: torch.cat(tensors, dim=1)
        self.after_concat = nn.Conv1d(embed_dim * 2, embed_dim // 2, 5, padding=2)
        self.last_conv = nn.Conv1d(embed_dim // 2, in_channel - 1, 5, padding=2)

    def _build_chunks(self):
        chunk_starts = list(range(0, self.seq_len, self.chunk_size))
        self.chunk_modules = nn.ModuleList()
        for cs in chunk_starts:
            ce = min(cs + self.chunk_size, self.seq_len)
            mask_start = max(0, cs - self.attention_range)
            mask_end = min(ce + self.attention_range, self.seq_len)
            module = ChunkModule(mask_end - mask_start, self.embed_dim, self.num_heads,
                                 start_offset=cs - mask_start, end_offset=mask_end - ce)
            self.chunk_modules.append(module)

    def forward(self, x):
        # x shape: (batch, seq_len, in_channel)
        x = self.embedding(x)
        chunks = []
        for module in self.chunk_modules:
            cs = module.transformer.start_offset
            ce = x.size(1) - module.transformer.end_offset
            chunk = module(x[:, cs:ce])
            chunks.append(chunk)
        x = self.concat(chunks)
        # conv layers expect (batch, channels, seq)
        x = self.after_concat(x.transpose(1, 2)).transpose(1, 2)
        x = self.last_conv(x.transpose(1, 2)).transpose(1, 2)
        return x[:, self.offset_before:self.seq_len - self.offset_after]
