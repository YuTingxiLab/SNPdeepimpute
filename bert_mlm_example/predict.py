import torch
from model import LongformerMLMVAE
from preprocess import read_vcf_haplotypes, build_onehot_dataset


def load_model(model_path, num_alleles, seq_len):
    model = LongformerMLMVAE(num_alleles=num_alleles, max_length=seq_len)
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.eval()
    return model


def fill_vcf(
    vcf_path,
    dataset_path='data/dataset.pt',
    output_path='filled.vcf',
    model_path='models/mlm_model.pt',
):
    haps, allele_values = read_vcf_haplotypes(vcf_path)
    data = torch.load(dataset_path)
    train_alleles = data['allele_values']
    inputs, _, _ = build_onehot_dataset(haps, train_alleles, min_mr=0.0, max_mr=0.0)
    mask = 1 - inputs[:, :, -1].long()
    model = load_model(
        model_path,
        num_alleles=len(train_alleles) + 1,
        seq_len=inputs.size(1),
    )
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
                a = train_alleles[preds[2*s, snp_idx].item()]
                b = train_alleles[preds[2*s+1, snp_idx].item()]
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
    p.add_argument('--dataset', default='data/dataset.pt')
    p.add_argument('--output', default='filled.vcf')
    p.add_argument('--model_path', default='models/mlm_model.pt')
    args = p.parse_args()
    fill_vcf(args.vcf, dataset_path=args.dataset, output_path=args.output, model_path=args.model_path)
