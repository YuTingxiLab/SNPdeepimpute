import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import SimpleTokenizer
from model import BertMLM


class MLMDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            data = json.load(f)
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long)
        }


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = torch.zeros_like(labels).bool()
    for special_id in [tokenizer.vocab[t] for t in ['[CLS]', '[SEP]', '[PAD]']]:
        special_tokens_mask |= inputs.eq(special_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    mask_id = tokenizer.vocab['[MASK]']
    inputs[masked_indices] = mask_id
    return inputs, labels


def train(data_path='data/dataset_vcf.json', tokenizer_path='data/tokenizer_vcf.json',
         model_out='models/mlm_model.pth', epochs=5, batch_size=4, lr=5e-4):
    dataset = MLMDataset(data_path)
    max_length = len(dataset.input_ids[0])
    tokenizer = SimpleTokenizer.load(tokenizer_path, max_length=max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BertMLM(vocab_size=len(tokenizer.vocab), max_length=max_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    def accuracy(logits, labels):
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct, total

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            inputs, labels = mask_tokens(inputs, tokenizer)
            logits = model(inputs, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            c, t = accuracy(logits, labels)
            train_correct += c
            train_total += t
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total else 0.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids']
                attention_mask = batch['attention_mask']
                inputs, labels = mask_tokens(inputs, tokenizer)
                logits = model(inputs, attention_mask=attention_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
                c, t = accuracy(logits, labels)
                val_correct += c
                val_total += t
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total else 0.0
        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
    torch.save(model.state_dict(), model_out)
    print(f"Model saved to {model_out}")


if __name__ == '__main__':
    train()
