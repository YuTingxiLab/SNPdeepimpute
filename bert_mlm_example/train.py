import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from model import LongformerMLMVAE


class SNPDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.input_ids = data['inputs'].long()
        self.labels = data['labels'].long()
        self.allele_values = data.get('allele_values', [])
        self.mask_idx = data.get('mask_idx', len(self.allele_values))
        self.vocab_size = len(self.allele_values) + 1
        self.num_classes = len(self.allele_values)
        self.seq_len = data.get('seq_len', self.input_ids.size(1))

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def vae_loss(logits, labels, mu, logvar, kl_weight=1.0):
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return ce + kl_weight * kl, ce, kl


def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean()


def train(
    data_path='data/dataset.pt',
    model_out='models/mlm_model.pt',
    epochs=5,
    batch_size=8,
    lr=5e-4,
    max_position_embeddings=None,
    kl_anneal_epochs=2,
):
    dataset = SNPDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    seq_len = dataset.seq_len
    model = LongformerMLMVAE(
        vocab_size=dataset.vocab_size,
        num_classes=dataset.num_classes,
        max_length=max_position_embeddings or seq_len,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    patience = 3
    wait = 0

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        tot_acc = 0.0
        kl_weight = min(1.0, (epoch + 1) / max(1, kl_anneal_epochs))
        for x, y in train_loader:
            mask = (x != dataset.mask_idx).long()
            logits, mu, logvar = model(x, attention_mask=mask)
            loss, ce, kl = vae_loss(logits, y, mu, logvar, kl_weight=kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_acc += accuracy(logits, y).item()
        train_loss = tot_loss / len(train_loader)
        train_acc = tot_acc / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                mask = (x != dataset.mask_idx).long()
                logits, mu, logvar = model(x, attention_mask=mask)
                loss, _, _ = vae_loss(logits, y, mu, logvar, kl_weight=1.0)
                val_loss += loss.item()
                val_acc += accuracy(logits, y).item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f'Saved model to {model_out}')


if __name__ == '__main__':
    train()
