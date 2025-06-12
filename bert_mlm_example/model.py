import torch
from torch import nn
from transformers import LongformerModel, LongformerConfig


class LongformerMLMVAE(nn.Module):
    """Longformer-based VAE for SNP imputation with token embeddings."""

    def __init__(
        self,
        vocab_size,
        num_classes,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_length=2048,
        attention_window=64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.token_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_length, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size)

        config = LongformerConfig(
            vocab_size=vocab_size,
            pad_token_id=0,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            attention_window=[attention_window] * num_layers,
        )
        self.encoder = LongformerModel(config)
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if input_ids.size(1) > self.max_length:
            raise ValueError(
                f"Input sequence length {input_ids.size(1)} exceeds max_length {self.max_length}"
            )
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), device=input_ids.device
            ).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(position_ids)
        x = self.dropout(self.norm(x))
        outputs = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        h = outputs.last_hidden_state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = self.decoder(z)
        return logits, mu, logvar
