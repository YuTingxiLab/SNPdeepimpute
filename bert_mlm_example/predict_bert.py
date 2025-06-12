import argparse
import torch
from bert_model import BertMLM
from tokenizer import SimpleTokenizer


def fill_mask(text, model_path='models/bert_mlm.pth', tokenizer_path='data/tokenizer.json', max_length=32):
    tok = SimpleTokenizer.load(tokenizer_path, max_length=max_length)
    ids, attn = tok.encode(text)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(attn, dtype=torch.long).unsqueeze(0)
    model = BertMLM(vocab_size=len(tok.vocab), max_length=max_length)
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    preds = logits.argmax(dim=-1)[0].tolist()
    tokens = []
    for pid in preds:
        tokens.append(tok.inv_vocab.get(pid, '[UNK]'))
    return ' '.join(tokens).replace(' [sep]', '').replace('[cls] ', '').strip()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Predict masked sentence with BERT')
    p.add_argument('sentence')
    p.add_argument('--model_path', default='models/bert_mlm.pth')
    p.add_argument('--tokenizer_path', default='data/tokenizer.json')
    p.add_argument('--max_length', type=int, default=32)
    args = p.parse_args()
    result = fill_mask(args.sentence, args.model_path, args.tokenizer_path, args.max_length)
    print(result)
