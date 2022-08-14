import re
import torch
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from utils.scrapper import get_song_info
from models.klue_bert import MyBertModel
from preprocessor import bert_tokenizer


def predictions(lyrics):
    input_ids, attention_mask = bert_tokenizer(lyrics)
    model = MyBertModel()
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    vectors = outputs.detach().cpu().numpy()
    return vectors


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def load_data():
    songs_info = pd.read_parquet("data/songs.parquet")
    song_vectors = torch.load("data/songs.pt")
    return songs_info, song_vectors


def recommends(song_id):
    songs_info, song_vectors = load_data()
    song = get_song_info(song_id)
    song_pub_date = np.round(int(song["public_date"][:4]), -1)
    vector = predictions(song["lyrics"])[0]
    cos_dict = {i: cos_sim(vector, c) for i, c in enumerate(song_vectors)}
    cos_dict = dict(sorted(cos_dict.items(), key=lambda item: item[1], reverse=True))
    idx = list(cos_dict.keys())
    songs_info = songs_info.iloc[idx]
    songs_info["score"] = list(cos_dict.values())
    songs_info = songs_info[songs_info["public_date"] >= song_pub_date][songs_info["genre"] == song["genre"]].iloc[:10]

    return songs_info.to_dict('records')
