import os
import random
import torch
import numpy as np


def read_vcf_haplotypes(path):
    """Read VCF and return haplotype array of shape (samples*2, snps)."""
    sample_names = []
    haplotypes1 = []
    with open(path) as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                header = line.strip().split('\t')
                sample_names = header[9:]
                haplotypes1 = [[] for _ in range(len(sample_names)*2)]
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
                haplotypes1[2*idx].append(int(a))
                haplotypes1[2*idx+1].append(int(b))
    return np.array(haplotypes1, dtype=np.int64)


def one_hot_encode(haps, mask_prob=0.15):
    """Return masked inputs and labels in one-hot form."""
    N, L = haps.shape
    inputs = np.zeros((N, L, 6), dtype=np.float32)
    labels = np.zeros((N, L, 5), dtype=np.float32)
    for i in range(N):
        for j in range(L):
            allele = haps[i, j]
            labels[i, j, allele] = 1.0
            if random.random() < mask_prob:
                inputs[i, j, 5] = 1.0
            else:
                inputs[i, j, allele] = 1.0
    return inputs, labels


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess VCF to one-hot dataset')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/dataset.pt')
    parser.add_argument('--mask_prob', type=float, default=0.15)
    args = parser.parse_args()

    haps = read_vcf_haplotypes(args.input)
    inputs, labels = one_hot_encode(haps, mask_prob=args.mask_prob)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'inputs': torch.tensor(inputs), 'labels': torch.tensor(labels)}, args.output)
    print(f'Saved dataset to {args.output} with shape {inputs.shape}')


if __name__ == '__main__':
    main()
