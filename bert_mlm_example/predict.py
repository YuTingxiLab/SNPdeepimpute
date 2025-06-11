import torch
from model import LongformerMLMVAE
from preprocess import read_vcf_haplotypes, one_hot_encode


def load_model(model_path, seq_len):
    model = LongformerMLMVAE(max_length=seq_len)
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.eval()
    return model


def fill_vcf(vcf_path, output_path='filled.vcf', model_path='models/mlm_model.pt'):
    haps = read_vcf_haplotypes(vcf_path)
    inputs, _ = one_hot_encode(haps, mask_prob=0.0)
    inputs = torch.tensor(inputs)
    mask = (inputs.sum(-1) != 0).long()
    model = load_model(model_path, seq_len=inputs.size(1))
    with torch.no_grad():
        logits, _, _ = model(inputs, attention_mask=mask)
    preds = logits.argmax(dim=-1)
    num_samples = haps.shape[0] // 2
    snp_idx = 0
    new_lines = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith('#'):
                new_lines.append(line.rstrip())
                continue
            fields = line.strip().split('\t')
            gts = []
            for s in range(num_samples):
                a = preds[2*s, snp_idx].item()
                b = preds[2*s+1, snp_idx].item()
                gts.append(f'{a}|{b}')
            new_lines.append('\t'.join(fields[:9] + gts))
            snp_idx += 1
    with open(output_path, 'w') as out:
        for l in new_lines:
            out.write(l + '\n')
    print(f'Wrote filled VCF to {output_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Fill masked VCF')
    p.add_argument('--vcf', required=True)
    p.add_argument('--output', default='filled.vcf')
    p.add_argument('--model_path', default='models/mlm_model.pt')
    args = p.parse_args()
    fill_vcf(args.vcf, args.output, args.model_path)
