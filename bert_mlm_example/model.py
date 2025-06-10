import torch
from torch import nn


class BertMLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 2, num_heads: int = 4,
                 max_length: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_length, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        x = self.token_emb(input_ids) + self.pos_emb(position_ids)
        x = x.transpose(0, 1)
        if attention_mask is not None:
            attn_mask = (1 - attention_mask).bool()
        else:
            attn_mask = None
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        x = x.transpose(0, 1)
        logits = self.mlm_head(x)
        return logits
