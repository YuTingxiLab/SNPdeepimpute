import torch
from torch import nn
from transformers import LongformerModel, LongformerConfig


class CatEmbeddings(nn.Module):
    """Embedding layer for one-hot encoded categorical SNP input."""

    def __init__(self, num_alleles: int, n_snps: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_alleles, embedding_dim))
        self.position_embedding = nn.Embedding(n_snps, embedding_dim)
        self.n_snps = n_snps

        nn.init.xavier_uniform_(self.embedding)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, A) - one-hot encoded SNP alleles
        assert x.dim() == 3, "Input must be of shape (batch, snps, alleles)"
        B, L, A = x.shape
        x = x.to(dtype=self.embedding.dtype)
        allele_emb = torch.einsum("bla,ad->bld", x, self.embedding)

        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        pos_emb = self.position_embedding(pos_ids)
        return allele_emb + pos_emb


class LongformerMLMVAE(nn.Module):
    """Longformer-based VAE for SNP imputation with one-hot SNP inputs."""

    def __init__(
        self,
        num_alleles,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_length=2048,
        attention_window=64,
        chunk_size=None,
    ):
        super().__init__()
        self.num_alleles = num_alleles
        self.num_classes = num_alleles - 1  # exclude mask channel
        self.embed = CatEmbeddings(num_alleles, max_length, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size)

        config = LongformerConfig(
            vocab_size=1,  # unused but required
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
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.num_classes),
        )
        self.max_length = max_length
        self.chunk_size = chunk_size

    def _forward_chunk(self, x, attention_mask):
        emb = self.dropout(self.norm(self.embed(x)))
        outputs = self.encoder(inputs_embeds=emb, attention_mask=attention_mask)
        h = outputs.last_hidden_state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = self.decoder(z)
        return logits, mu, logvar

    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)
        if seq_len > self.max_length and self.chunk_size is None:
            raise ValueError(
                f"Input length {seq_len} exceeds max_length {self.max_length} and no chunk_size provided"
            )
        if self.chunk_size is None or seq_len <= self.chunk_size:
            if attention_mask is None:
                attention_mask = torch.ones(x.size(0), seq_len, device=x.device, dtype=torch.long)
            return self._forward_chunk(x, attention_mask)

        logits_list = []
        mu_list = []
        logvar_list = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            x_chunk = x[:, start:end, :]
            mask_chunk = (
                attention_mask[:, start:end] if attention_mask is not None else None
            )
            out_l, out_mu, out_logvar = self._forward_chunk(x_chunk, mask_chunk)
            logits_list.append(out_l)
            mu_list.append(out_mu)
            logvar_list.append(out_logvar)
        logits = torch.cat(logits_list, dim=1)
        mu = torch.cat(mu_list, dim=1)
        logvar = torch.cat(logvar_list, dim=1)
        return logits, mu, logvar
