import json
import os
from collections import Counter

SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']


class SimpleTokenizer:
    def __init__(self, vocab=None, max_length=32):
        self.max_length = max_length
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}

    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def encode(self, text):
        tokens = text.lower().split()
        ids = [self.vocab['[CLS]']]
        for tok in tokens:
            ids.append(self.vocab.get(tok, self.vocab['[MASK]']))
        ids.append(self.vocab['[SEP]'])
        if len(ids) < self.max_length:
            ids += [self.vocab['[PAD]']] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        attention = [1 if id != self.vocab['[PAD]'] else 0 for id in ids]
        return ids, attention

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path, max_length=32):
        with open(path) as f:
            vocab = json.load(f)
        tokenizer = cls(vocab=vocab, max_length=max_length)
        tokenizer.inv_vocab = {idx: tok for tok, idx in vocab.items()}
        return tokenizer
