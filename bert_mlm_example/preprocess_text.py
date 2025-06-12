import os
import json
import argparse
from tokenizer import SimpleTokenizer


def main():
    parser = argparse.ArgumentParser(description="Preprocess text for BERT MLM")
    parser.add_argument("--input", required=True, help="text file with one sentence per line")
    parser.add_argument("--output", default="data/dataset.json")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--max_length", type=int, default=32)
    args = parser.parse_args()

    with open(args.input) as f:
        texts = [line.strip() for line in f if line.strip()]

    tok = SimpleTokenizer(max_length=args.max_length)
    tok.build_vocab(texts)

    ids = []
    attn = []
    for text in texts:
        i, a = tok.encode(text)
        ids.append(i)
        attn.append(a)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.tokenizer), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"input_ids": ids, "attention_mask": attn}, f)
    tok.save(args.tokenizer)
    print(f"Saved dataset to {args.output} with {len(ids)} samples")


if __name__ == "__main__":
    main()
