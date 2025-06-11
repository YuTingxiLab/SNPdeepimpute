import json
import torch
from tokenizer import SimpleTokenizer
from model import BertMLM


def _load_model(model_path, tokenizer_path, data_path):
    with open(data_path) as f:
        max_length = len(json.load(f)['input_ids'][0])
    tokenizer = SimpleTokenizer.load(tokenizer_path, max_length=max_length)
    model = BertMLM(vocab_size=len(tokenizer.vocab), max_length=max_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, tokenizer


def fill_mask(sentence: str, model_path='models/mlm_model.pth', tokenizer_path='data/tokenizer_vcf.json',
              data_path='data/dataset_vcf.json'):
    model, tokenizer = _load_model(model_path, tokenizer_path, data_path)
    ids, attention = tokenizer.encode(sentence)
    input_ids = torch.tensor([ids])
    attention_mask = torch.tensor([attention])
    mask_index = input_ids.eq(tokenizer.vocab['[MASK]']).nonzero(as_tuple=True)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    tokens = sentence.split()
    for idx in mask_index[1]:
        pred_id = logits[0, idx, :].argmax(dim=-1).item()
        tokens[idx - 1] = tokenizer.inv_vocab[pred_id]
    print(' '.join(tokens))


def fill_vcf(vcf_path: str, output_path: str, model_path='models/mlm_model.pth',
             tokenizer_path='data/tokenizer_vcf.json', data_path='data/dataset_vcf.json'):
    model, tokenizer = _load_model(model_path, tokenizer_path, data_path)

    header = []
    sample_names = []
    genos = {}
    other_fields = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith('##'):
                header.append(line.rstrip())
                continue
            if line.startswith('#CHROM'):
                header.append(line.rstrip())
                hdr = line.strip().split('\t')
                sample_names = hdr[9:]
                genos = {name: [] for name in sample_names}
                continue
            fields = line.strip().split('\t')
            gts = fields[9:]
            for name, gt in zip(sample_names, gts):
                token = gt.split(':')[0]
                if token == '.' or token == '[MASK]':
                    token = '[MASK]'
                genos[name].append(token)
            other_fields.append((fields[:9], gts))

    # predict for each sample
    for name in sample_names:
        tokens = genos[name]
        sentence = ' '.join(tokens)
        ids, attn = tokenizer.encode(sentence)
        input_ids = torch.tensor([ids])
        attention_mask = torch.tensor([attn])
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)
        preds = logits[0].argmax(dim=-1)
        for i, tok in enumerate(tokens):
            if tok == '[MASK]':
                token_id = preds[i + 1].item()
                tokens[i] = tokenizer.inv_vocab[token_id]
        genos[name] = tokens

    # write output VCF
    with open(output_path, 'w') as out:
        for l in header:
            out.write(l + '\n')
        for idx, (pre_fields, orig_gts) in enumerate(other_fields):
            new_fields = pre_fields[:]
            for name, orig in zip(sample_names, orig_gts):
                suffix = orig.split(':')[1:]
                gt = genos[name][idx]
                if suffix:
                    new_fields.append(gt + ':' + ':'.join(suffix))
                else:
                    new_fields.append(gt)
            out.write('\t'.join(new_fields) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fill masks in text or VCF')
    parser.add_argument('--sentence', help='Sentence with [MASK] token')
    parser.add_argument('--vcf', help='VCF file with masked genotypes')
    parser.add_argument('--output', default='filled.vcf', help='Output VCF path')
    parser.add_argument('--model_path', default='models/mlm_model.pth')
    parser.add_argument('--tokenizer_path', default='data/tokenizer_vcf.json')
    parser.add_argument('--data_path', default='data/dataset_vcf.json')
    args = parser.parse_args()

    if args.vcf:
        fill_vcf(args.vcf, args.output, args.model_path, args.tokenizer_path, args.data_path)
    else:
        sentence = args.sentence or '0|0 [MASK] 1|0'
        fill_mask(sentence, args.model_path, args.tokenizer_path, args.data_path)
