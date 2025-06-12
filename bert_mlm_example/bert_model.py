import torch
from torch import nn


class BertMLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, num_layers=2, num_heads=4, max_length=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_length, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        if L > self.max_length:
            raise ValueError(f"sequence length {L} exceeds model max_length {self.max_length}")
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos_ids)
        if attention_mask is not None:
            # Transformer expects mask with True for positions to skip
            attn_mask = attention_mask == 0
        else:
            attn_mask = None
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        x = self.norm(x)
        logits = self.mlm_head(x)
        return logits
