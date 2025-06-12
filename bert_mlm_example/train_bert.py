import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from bert_model import BertMLM
from tokenizer import SimpleTokenizer
import torch.nn.functional as F


class TextDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            data = json.load(f)
        self.input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(data["attention_mask"], dtype=torch.long)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]


def mask_inputs(inputs, mask_token_id, pad_token_id, mlm_prob=0.15):
    labels = inputs.clone()
    prob = torch.full(labels.shape, mlm_prob)
    special = inputs.eq(pad_token_id)
    prob.masked_fill_(special, 0.0)
    masked_indices = torch.bernoulli(prob).bool()
    inputs = inputs.clone()
    inputs[masked_indices] = mask_token_id
    labels[~masked_indices] = -100
    return inputs, labels


def train(data_path='data/dataset.json', tokenizer_path='data/tokenizer.json', model_out='models/bert_mlm.pth', epochs=5, batch_size=8, lr=5e-4):
    dataset = TextDataset(data_path)
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BertMLM(vocab_size=len(tokenizer.vocab), max_length=dataset.input_ids.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mask_id = tokenizer.vocab['[MASK]']
    pad_id = tokenizer.vocab['[PAD]']

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        tot_correct = 0
        tot_tokens = 0
        for ids, attn in loader:
            inputs, labels = mask_inputs(ids, mask_id, pad_id)
            logits = model(inputs, attention_mask=attn)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            tot_correct += (preds[mask] == labels[mask]).float().sum().item()
            tot_tokens += mask.sum().item()
        acc = tot_correct / max(1, tot_tokens)
        print(f"Epoch {epoch+1} - Loss: {tot_loss/len(loader):.4f} Acc: {acc:.4f}")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"Saved model to {model_out}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Train BERT for MLM')
    p.add_argument('--data_path', default='data/dataset.json')
    p.add_argument('--tokenizer_path', default='data/tokenizer.json')
    p.add_argument('--model_out', default='models/bert_mlm.pth')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    args = p.parse_args()
    train(data_path=args.data_path, tokenizer_path=args.tokenizer_path, model_out=args.model_out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
