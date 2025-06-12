import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    """Standard VAE loss with optional R2 term for imputation."""

    def __init__(self, beta: float = 1.0, use_r2_loss: bool = True, group_size: int = 4):
        super().__init__()
        self.beta = beta
        self.use_r2_loss = use_r2_loss
        self.group_size = group_size

    def _minimac_r2(self, pred_alt: torch.Tensor, gt_alt_af: torch.Tensor) -> torch.Tensor:
        mask = (gt_alt_af == 0.0) | (gt_alt_af == 1.0)
        gt_alt_af = torch.where(mask, torch.full_like(gt_alt_af, 0.5), gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = torch.where(denom < 0.01, torch.full_like(denom, 0.01), denom)
        r2 = torch.mean((pred_alt - gt_alt_af) ** 2, dim=0) / denom
        r2 = torch.where(mask, torch.zeros_like(r2), r2)
        return r2

    def forward(self, logits, labels_onehot, mu, logvar, beta: float = None):
        target = labels_onehot.argmax(dim=-1)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        b = self.beta if beta is None else beta
        total_loss = ce + b * kl

        if self.use_r2_loss:
            probs = logits.softmax(dim=-1)
            batch_size, seq_len, _ = probs.shape
            group = self.group_size
            num_full = batch_size // group
            r2_loss = 0.0

            for i in range(num_full):
                pred_group = probs[i * group : (i + 1) * group]
                true_group = labels_onehot[i * group : (i + 1) * group]
                gt_alt_af = true_group.argmax(dim=-1).float().sum(dim=0) / group
                pred_alt_prob = pred_group[..., 1:].sum(dim=-1).mean(dim=0)
                r2_loss += -self._minimac_r2(pred_alt_prob, gt_alt_af).sum() * group

            if batch_size % group != 0:
                start = num_full * group
                pred_rem = probs[start:]
                true_rem = labels_onehot[start:]
                n = pred_rem.size(0)
                gt_alt_af = true_rem.argmax(dim=-1).float().sum(dim=0) / n
                pred_alt_prob = pred_rem[..., 1:].sum(dim=-1).mean(dim=0)
                r2_loss += -self._minimac_r2(pred_alt_prob, gt_alt_af).sum() * n

            total_loss = total_loss + r2_loss / (batch_size * seq_len)

        return total_loss, ce, kl
