# BERT MLM Example with SNP VCF Data

This directory contains a minimal example for training a BERT-style model to perform masked language modelling (MLM) on genotype data. A synthetic VCF file with 100 samples and 1000 SNPs is provided under `data/synthetic_100x1000.vcf`.

Directory layout:

- `data/` – contains VCF data, tokenizers and preprocessed datasets
- `models/` – stores trained model weights

## Preprocessing

Convert the VCF file to token ids and build a vocabulary. The example below
uses a maximum sequence length of 1024 so that each sample encodes all 1000
SNPs:

```bash
python preprocess.py --input data/synthetic_100x1000.vcf --output data/dataset_vcf.json --tokenizer data/tokenizer_vcf.json --max_length 1024
```

## Training

Train the model using the preprocessed dataset. The training script will
automatically determine the correct positional embedding size from the dataset:

```bash
python train.py --epochs 2 --batch_size 2
```

The script saves the weights to `models/mlm_model.pth`.

## Prediction

Fill a masked genotype token using the trained model. The prediction script also
reads the dataset to determine the required sequence length:

```bash
python predict.py "0|0 [MASK] 1|0"
```
