import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    """VAE loss combining reconstruction CE and KL divergence."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels_onehot, mu, logvar, kl_weight=1.0):
        target = labels_onehot.argmax(dim=-1)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return ce + kl_weight * kl, ce, kl
