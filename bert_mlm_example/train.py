import json
import torch
from torch.utils.data import Dataset, DataLoader
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertMLM(vocab_size=len(tokenizer.vocab), max_length=max_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            inputs, labels = mask_tokens(inputs, tokenizer)
            logits = model(inputs, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), model_out)
    print(f"Model saved to {model_out}")


if __name__ == '__main__':
    train()
