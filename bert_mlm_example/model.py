import torch
from torch import nn
from transformers import LongformerModel, LongformerConfig
from .rmsnorm import RMSNorm, RMSNormTransformerEncoderLayer


class RelPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding that generalizes to long sequences."""

    def __init__(self, embedding_dim: int, max_length: int = 4096):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        sinusoid_inp = torch.outer(t, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if pos_emb.size(-1) != self.embedding_dim:
            pos_emb = torch.cat([pos_emb, pos_emb.new_zeros(seq_len, self.embedding_dim - pos_emb.size(-1))], dim=-1)
        return pos_emb


class CatEmbeddings(nn.Module):
    """Embedding layer for one-hot encoded SNPs with sinusoidal relative positions."""

    def __init__(self, num_alleles: int, embedding_dim: int, max_length: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_alleles, embedding_dim))
        self.pos_encoder = RelPositionalEncoding(embedding_dim, max_length)

        nn.init.xavier_uniform_(self.embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, A) - one-hot encoded SNP alleles
        assert x.dim() == 3, "Input must be of shape (batch, snps, alleles)"
        B, L, A = x.shape
        x = x.to(dtype=self.embedding.dtype)
        allele_emb = torch.einsum("bla,ad->bld", x, self.embedding)

        pos_emb = self.pos_encoder(L)
        pos_emb = pos_emb.unsqueeze(0).expand(B, L, -1)
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
        latent_size=None,
        overlap=64,
    ):
        super().__init__()
        self.num_alleles = num_alleles
        self.num_classes = num_alleles - 1  # exclude mask channel
        self.embed = CatEmbeddings(num_alleles, hidden_size, max_length)
        self.dropout = nn.Dropout(0.1)
        self.norm = RMSNorm(hidden_size, eps=1e-8)

        self.latent_size = latent_size or hidden_size // 2

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

        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.anchor_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.global_token, std=0.02)
        nn.init.normal_(self.anchor_token, std=0.02)
        self.global_interactor = nn.TransformerEncoder(
            RMSNormTransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, self.latent_size)
        self.fc_logvar = nn.Linear(hidden_size, self.latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, hidden_size),
            nn.ReLU(),
            RMSNorm(hidden_size, eps=1e-8),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            RMSNorm(hidden_size, eps=1e-8),
            nn.Linear(hidden_size, self.num_classes),
        )
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.overlap = overlap

    def encode_chunk(self, x, attention_mask=None):
        B, L, _ = x.shape
        emb = self.dropout(self.norm(self.embed(x)))
        g = self.global_token.expand(B, 1, -1)
        emb = torch.cat([g, emb], dim=1)
        if attention_mask is not None:
            mask = torch.cat(
                [torch.ones(B, 1, device=x.device, dtype=torch.long), attention_mask],
                dim=1,
            )
        else:
            mask = torch.ones(B, L + 1, device=x.device, dtype=torch.long)
        outputs = self.encoder(inputs_embeds=emb, attention_mask=mask)
        h = outputs.last_hidden_state
        return h[:, 1:], h[:, :1]

    def decode_tokens(self, tokens):
        mu = self.fc_mu(tokens)
        logvar = self.fc_logvar(tokens)
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
            tokens, g = self.encode_chunk(x, attention_mask)
            g = self.global_interactor(torch.cat([self.anchor_token.expand(x.size(0), 1, -1), g], dim=1))[:, 1:]
            attn_out, _ = self.cross_attn(tokens, g, g)
            tokens = tokens + attn_out
            return self.decode_tokens(tokens)

        stride = self.chunk_size - self.overlap
        if stride <= 0:
            raise ValueError("chunk_size must be larger than overlap")

        chunks = []
        globals_list = []
        starts = []
        for start in range(0, seq_len, stride):
            end = min(start + self.chunk_size, seq_len)
            chunk_x = x[:, start:end, :]
            mask_chunk = attention_mask[:, start:end] if attention_mask is not None else None
            t, g = self.encode_chunk(chunk_x, mask_chunk)
            chunks.append(t)
            globals_list.append(g)
            starts.append(start)
            if end == seq_len:
                break

        globals_cat = torch.cat(globals_list, dim=1)
        globals_updated = self.global_interactor(
            torch.cat([self.anchor_token.expand(x.size(0), 1, -1), globals_cat], dim=1)
        )[:, 1:]

        out_logits = torch.zeros(x.size(0), seq_len, self.num_classes, device=x.device)
        out_mu = torch.zeros(x.size(0), seq_len, self.latent_size, device=x.device)
        out_logvar = torch.zeros(x.size(0), seq_len, self.latent_size, device=x.device)
        counts = torch.zeros(seq_len, device=x.device)

        for i, (tokens, start) in enumerate(zip(chunks, starts)):
            g = globals_updated[:, i : i + 1]
            attn_out, _ = self.cross_attn(tokens, g, g)
            tokens_fused = tokens + attn_out
            logits, mu, logvar = self.decode_tokens(tokens_fused)
            end = min(start + tokens.size(1), seq_len)
            out_logits[:, start:end] += logits[:, : end - start]
            out_mu[:, start:end] += mu[:, : end - start]
            out_logvar[:, start:end] += logvar[:, : end - start]
            counts[start:end] += 1

        counts = counts.clamp_min(1.0)
        out_logits = out_logits / counts.unsqueeze(0).unsqueeze(-1)
        out_mu = out_mu / counts.unsqueeze(0).unsqueeze(-1)
        out_logvar = out_logvar / counts.unsqueeze(0).unsqueeze(-1)
        return out_logits, out_mu, out_logvar
