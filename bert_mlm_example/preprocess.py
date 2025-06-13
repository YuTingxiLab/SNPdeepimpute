import os
import random
import torch
import numpy as np
import torch.nn.functional as F


def read_vcf_haplotypes(path):
    """Read VCF and return haplotype array and encountered allele values."""
    sample_names = []
    haplotypes = []
    allele_set = set()
    with open(path) as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                header = line.strip().split('\t')
                sample_names = header[9:]
                haplotypes = [[] for _ in range(len(sample_names) * 2)]
                continue
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            gts = fields[9:]
            for idx, gt in enumerate(gts):
                gt = gt.split(':')[0]
                if '|' in gt:
                    a, b = gt.split('|')
                else:
                    a, b = gt.split('/')
                a = int(a) if a.isdigit() else 0
                b = int(b) if b.isdigit() else 0
                haplotypes[2 * idx].append(a)
                haplotypes[2 * idx + 1].append(b)
                allele_set.add(a)
                allele_set.add(b)
    allele_values = sorted(allele_set)
    return np.array(haplotypes, dtype=np.int64), allele_values


def add_attention_mask(x_sample, y_sample, depth, min_mr, max_mr):
    """Mask random positions and return one-hot encoded x and y."""
    if not torch.is_tensor(x_sample):
        x_sample = torch.tensor(x_sample).long()
    if not torch.is_tensor(y_sample):
        y_sample = torch.tensor(y_sample).long()

    seq_len = len(x_sample)
    masking_rate = torch.empty(1).uniform_(min_mr, max_mr).item()
    mask_size = int(seq_len * masking_rate)

    mask_idx = torch.randperm(seq_len)[:mask_size]

    x_masked = x_sample.clone()
    x_masked[mask_idx] = depth - 1

    x_onehot = F.one_hot(x_masked, num_classes=depth).float()
    y_onehot = F.one_hot(y_sample, num_classes=depth - 1).float()

    return x_onehot, y_onehot, mask_idx


def build_onehot_dataset(haps, allele_values, min_mr=0.05, max_mr=0.15):
    """Return masked one-hot inputs and labels."""
    allele_to_idx = {v: i for i, v in enumerate(allele_values)}
    depth = len(allele_values) + 1
    N, L = haps.shape
    inputs = torch.zeros(N, L, depth, dtype=torch.float32)
    labels = torch.zeros(N, L, depth - 1, dtype=torch.float32)
    for i in range(N):
        ids = [allele_to_idx[a] for a in haps[i]]
        x_onehot, y_onehot, _ = add_attention_mask(ids, ids, depth, min_mr, max_mr)
        inputs[i] = x_onehot
        labels[i] = y_onehot
    return inputs, labels, depth


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess VCF to one-hot dataset')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/dataset.pt')
    parser.add_argument('--min_mask_rate', type=float, default=0.05)
    parser.add_argument('--max_mask_rate', type=float, default=0.15)
    args = parser.parse_args()

    haps, allele_values = read_vcf_haplotypes(args.input)
    inputs, labels, depth = build_onehot_dataset(
        haps,
        allele_values,
        min_mr=args.min_mask_rate,
        max_mr=args.max_mask_rate,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'inputs': inputs,
        'labels': labels,
        'allele_values': allele_values,
        'seq_len': inputs.shape[1],
        'num_alleles': depth,
    }, args.output)
    print(
        f'Saved dataset to {args.output} with shape {inputs.shape} and '
        f'{len(allele_values)} allele types'
    )


if __name__ == '__main__':
    main()
