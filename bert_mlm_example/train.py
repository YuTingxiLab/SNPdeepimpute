import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from model import LongformerMLMVAE


class SNPDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.inputs = data['inputs']
        self.labels = data['labels']

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def vae_loss(logits, labels, mu, logvar):
    target = labels.argmax(dim=-1)
    ce = F.cross_entropy(logits.view(-1, 5), target.view(-1))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return ce + kl, ce, kl


def train(data_path='data/dataset.pt', model_out='models/mlm_model.pt', epochs=5,
          batch_size=8, lr=5e-4):
    dataset = SNPDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    seq_len = dataset.inputs.size(1)
    model = LongformerMLMVAE(max_length=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        for x, y in train_loader:
            mask = (x.sum(-1) != 0).long()
            logits, mu, logvar = model(x, attention_mask=mask)
            loss, ce, kl = vae_loss(logits, y, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        train_loss = tot_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                mask = (x.sum(-1) != 0).long()
                logits, mu, logvar = model(x, attention_mask=mask)
                loss, _, _ = vae_loss(logits, y, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f'Saved model to {model_out}')


if __name__ == '__main__':
    train()
