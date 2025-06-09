from typing import Tuple
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .model import DeepImpute
from .losses import ImputationLoss


def add_attention_mask(x: torch.Tensor, depth: int, min_mr: float, max_mr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = x.size(0)
    masking_rate = torch.empty(1).uniform_(min_mr, max_mr).item()
    mask_size = int(seq_len * masking_rate)
    mask_idx = torch.randperm(seq_len)[:mask_size]
    x_masked = x.clone()
    x_masked[mask_idx] = depth - 1
    return nn.functional.one_hot(x_masked, depth), nn.functional.one_hot(x, depth - 1)


def onehot_encode(x: torch.Tensor, depth: int) -> torch.Tensor:
    return nn.functional.one_hot(x, depth)


class GenotypeDataset(Dataset):
    """Simple dataset wrapper for genotype arrays."""

    def __init__(self, data: torch.Tensor, depth: int, offset_before: int, offset_after: int,
                 training: bool = True, masking_rates=(0.5, 0.99)):
        self.data = data
        self.depth = depth
        self.offset_before = offset_before
        self.offset_after = offset_after
        self.training = training
        self.min_mr, self.max_mr = masking_rates

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = sample[self.offset_before: sample.size(0) - self.offset_after]
        if self.training:
            x, y = add_attention_mask(sample, self.depth, self.min_mr, self.max_mr)
            return x.float(), y.float()
        else:
            x = onehot_encode(sample, self.depth)
            y = onehot_encode(target, self.depth - 1)
            return x.float(), y.float()


def create_model(seq_len: int, depth: int, args) -> nn.Module:
    model = DeepImpute(seq_len=seq_len, in_channel=depth, embed_dim=args["embedding_dim"],
                  num_heads=args["num_heads"], chunk_size=args["chunk_size"],
                  attention_range=args["chunk_overlap"], offset_before=args["offset_before"],
                  offset_after=args["offset_after"])
    return model


def train(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)
