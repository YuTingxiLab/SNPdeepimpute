import torch
from torch import nn
import torch.nn.functional as F


class ImputationLoss(nn.Module):
    """Combination of cross entropy, KL divergence and optional r2 loss."""

    def __init__(self, use_r2_loss: bool = True):
        super().__init__()
        self.use_r2_loss = use_r2_loss
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.kl = nn.KLDivLoss(reduction='sum')

    @staticmethod
    def _minimac_r2(pred_alt_probs, gt_alt_af):
        mask = (gt_alt_af == 0.0) | (gt_alt_af == 1.0)
        gt_alt_af = torch.where(mask, torch.full_like(gt_alt_af, 0.5), gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = torch.where(denom < 0.01, torch.full_like(denom, 0.01), denom)
        r2 = torch.mean((pred_alt_probs - gt_alt_af) ** 2, dim=0) / denom
        r2 = torch.where(mask, torch.zeros_like(r2), r2)
        return r2

    def forward(self, y_pred, y_true):
        # y_true: one-hot encoded (batch, seq, C)
        log_pred = torch.log(torch.clamp(y_pred, min=1e-7))
        ce_loss = self.ce(log_pred.transpose(1, 2), y_true.argmax(dim=-1))
        kl_loss = self.kl(log_pred, y_true)
        total = ce_loss + kl_loss

        if self.use_r2_loss:
            batch_size, seq_len, _ = y_pred.shape
            group_size = 4
            num_full = batch_size // group_size
            num_rem = batch_size % group_size
            r2_loss = 0.0
            for i in range(num_full):
                gt_group = y_true[i*group_size:(i+1)*group_size]
                pred_group = y_pred[i*group_size:(i+1)*group_size]
                gt_alt_af = gt_group.argmax(dim=-1).float().sum(0) / group_size
                pred_alt = pred_group[:, :, 1:].sum(-1)
                r2_loss += -self._minimac_r2(pred_alt, gt_alt_af).sum() * group_size
            if num_rem > 0:
                gt_group = y_true[-num_rem:]
                pred_group = y_pred[-num_rem:]
                gt_alt_af = gt_group.argmax(dim=-1).float().sum(0) / num_rem
                pred_alt = pred_group[:, :, 1:].sum(-1)
                r2_loss += -self._minimac_r2(pred_alt, gt_alt_af).sum() * num_rem
            total += r2_loss
        return total
