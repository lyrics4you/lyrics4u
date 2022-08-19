import torch
import pandas as pd

def bert_tokenizer(tokenizer, sentence, max_len=512):
    encoded = tokenizer.encode_plus(
        text=sentence,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    return torch.tensor([encoded["input_ids"]], dtype=torch.long), torch.tensor(
        torch.tensor([encoded["attention_mask"]], dtype=torch.long)
    )


def load_data():
    songs_info = pd.read_csv("data/songs.tsv", sep="\t")
    song_vectors = torch.load("data/songs3.pt")
    emotion_vectors =  torch.load('data/emo_logit_vertors.pt')
    return songs_info, song_vectors, emotion_vectors