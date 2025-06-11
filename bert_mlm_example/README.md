# Longformer VAE for SNP Imputation

This example demonstrates training a Longformer based model to fill masked SNP genotypes. A synthetic VCF containing **1000 samples** each with **1000 SNPs** is provided at `data/synthetic_1000x1000.vcf`.

Directory layout:

- `data/` – VCF data and preprocessed tensors
- `models/` – trained model weights

## Preprocessing

Convert the VCF to a one‑hot encoded tensor dataset. Each diploid genotype is split into haplotypes and randomly masked. The resulting tensor has shape `[2000, 1000, 6]`.

```bash
python preprocess.py --input data/synthetic_1000x1000.vcf --output data/dataset.pt
```

## Training

Pretrain the model on the masked dataset. The model outputs logits of shape `[batch, seq_len, 5]` and is optimised with a VAE‑style loss (cross entropy reconstruction plus KL term).

```bash
python train.py --epochs 2 --batch_size 8 --data_path data/dataset.pt
```

## Prediction

Fill an incomplete VCF using the trained model:

```bash
python predict.py --vcf data/synthetic_1000x1000.vcf --output data/filled.vcf
```

The output file will contain genotypes for all samples with masked sites completed.
