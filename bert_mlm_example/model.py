import torch
from torch import nn
from transformers import LongformerModel, LongformerConfig


class LongformerMLMVAE(nn.Module):
    def __init__(self, input_dim=6, hidden_size=128, num_layers=2, num_heads=4,
                 max_length=1000, attention_window=64):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_size)
        config = LongformerConfig(
            vocab_size=1,
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
        self.decoder = nn.Linear(hidden_size, 5)

    def forward(self, inputs, attention_mask=None):
        # inputs: [batch, seq_len, 6]
        x = self.embed(inputs)
        outputs = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        h = outputs.last_hidden_state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = self.decoder(z)
        return logits, mu, logvar
