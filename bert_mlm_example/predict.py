import torch
from tokenizer import SimpleTokenizer
from model import BertMLM


def fill_mask(sentence: str, model_path='models/mlm_model.pth', tokenizer_path='data/tokenizer_vcf.json', max_length: int = 32):
    tokenizer = SimpleTokenizer.load(tokenizer_path, max_length=max_length)
    model = BertMLM(vocab_size=len(tokenizer.vocab))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    ids, attention = tokenizer.encode(sentence)
    input_ids = torch.tensor([ids])
    attention_mask = torch.tensor([attention])
    mask_index = input_ids.eq(tokenizer.vocab['[MASK]']).nonzero(as_tuple=True)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    mask_logits = logits[0, mask_index[1], :]
    predicted_id = mask_logits.argmax(dim=-1).item()
    predicted_token = tokenizer.inv_vocab[predicted_id]
    tokens = sentence.split()
    output_tokens = [predicted_token if t == '[MASK]' else t for t in tokens]
    print(' '.join(output_tokens))


if __name__ == '__main__':
    import sys
    sentence = sys.argv[1] if len(sys.argv) > 1 else '0|0 [MASK] 1|0'
    fill_mask(sentence)
