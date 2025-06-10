import json
from tokenizer import SimpleTokenizer


def read_vcf(path):
    """Extract genotype sequences from a VCF file."""
    samples = []
    sample_names = []
    genos = {}
    with open(path) as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                header = line.strip().split('\t')
                sample_names = header[9:]
                genos = {name: [] for name in sample_names}
                continue
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            gts = fields[9:]
            for name, gt in zip(sample_names, gts):
                genos[name].append(gt.split(':')[0])
    for name in sample_names:
        samples.append(' '.join(genos[name]))
    return samples


def read_corpus(path):
    if path.endswith('.vcf'):
        return read_vcf(path)
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess text for simple BERT MLM')
    parser.add_argument('--input', required=True)
    parser.add_argument('--tokenizer', default='data/tokenizer.json')
    parser.add_argument('--output', default='data/dataset.json')
    parser.add_argument('--max_length', type=int, default=32)
    args = parser.parse_args()

    sentences = read_corpus(args.input)
    tokenizer = SimpleTokenizer(max_length=args.max_length)
    tokenizer.build_vocab(sentences)
    tokenizer.save(args.tokenizer)
    input_ids, attention_masks = [], []
    for s in sentences:
        ids, attn = tokenizer.encode(s)
        input_ids.append(ids)
        attention_masks.append(attn)
    with open(args.output, 'w') as f:
        json.dump({'input_ids': input_ids, 'attention_mask': attention_masks}, f)
    print(f"Saved dataset to {args.output} with vocab size {len(tokenizer.vocab)}")


if __name__ == '__main__':
    main()
