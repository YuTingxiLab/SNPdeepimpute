import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model import LongformerMLMVAE
from losses import VAELoss


class SNPDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.inputs = data['inputs'].float()
        self.labels = data['labels'].float()
        self.allele_values = data.get('allele_values', [])
        self.num_alleles = data.get('num_alleles', self.inputs.size(-1))
        self.num_classes = self.num_alleles - 1
        self.seq_len = data.get('seq_len', self.inputs.size(1))

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    targets = labels.argmax(dim=-1)
    return (preds == targets).float().mean()


def train(
    data_path='data/dataset.pt',
    model_out='models/mlm_model.pt',
    epochs=5,
    batch_size=8,
    lr=5e-4,
    max_position_embeddings=None,
    kl_anneal_epochs=2,
    chunk_size=None,
):
    dataset = SNPDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    seq_len = dataset.seq_len
    model = LongformerMLMVAE(
        num_alleles=dataset.num_alleles,
        max_length=max_position_embeddings or seq_len,
        chunk_size=chunk_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = VAELoss()

    best_val = float('inf')
    patience = 3
    wait = 0

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        tot_acc = 0.0
        kl_weight = min(1.0, (epoch + 1) / max(1, kl_anneal_epochs))
        for x, y in train_loader:
            mask = 1 - x[:, :, -1].long()
            logits, mu, logvar = model(x, attention_mask=mask)
            loss, ce, kl = criterion(logits, y, mu, logvar, kl_weight=kl_weight)
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
                mask = 1 - x[:, :, -1].long()
                logits, mu, logvar = model(x, attention_mask=mask)
                loss, _, _ = criterion(logits, y, mu, logvar, kl_weight=1.0)
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
