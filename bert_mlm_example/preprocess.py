import os
import random
import torch
import numpy as np


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


def build_token_dataset(haps, allele_values, mask_prob=0.15):
    """Return masked input IDs and label IDs."""
    allele_to_idx = {v: i for i, v in enumerate(allele_values)}
    mask_idx = len(allele_values)
    num_classes = len(allele_values)
    N, L = haps.shape
    inputs = np.full((N, L), mask_idx, dtype=np.int64)
    labels = np.zeros((N, L), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            allele = haps[i, j]
            idx = allele_to_idx[allele]
            labels[i, j] = idx
            if random.random() >= mask_prob:
                inputs[i, j] = idx
    return inputs, labels, mask_idx


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess VCF to token dataset')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/dataset.pt')
    parser.add_argument('--mask_prob', type=float, default=0.15)
    args = parser.parse_args()

    haps, allele_values = read_vcf_haplotypes(args.input)
    inputs, labels, mask_idx = build_token_dataset(
        haps, allele_values, mask_prob=args.mask_prob
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'inputs': torch.tensor(inputs, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'allele_values': allele_values,
        'mask_idx': mask_idx,
        'seq_len': inputs.shape[1]
    }, args.output)
    print(
        f'Saved dataset to {args.output} with shape {inputs.shape} and '
        f'{len(allele_values)} allele types (mask index {mask_idx})'
    )


if __name__ == '__main__':
    main()
