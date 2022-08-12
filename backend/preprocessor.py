from transformers import BertTokenizer
import torch


def bert_tokenizer(sentence, name="klue/bert-base", max_len=512):
    tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=False)

    encoded = tokenizer.encode_plus(
        text=sentence,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    return torch.tensor([encoded["input_ids"]], dtype=torch.long), torch.tensor(torch.tensor([encoded["attention_mask"]], dtype=torch.long))



