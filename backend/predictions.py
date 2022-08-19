import torch
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from utils.scrapper import get_song_info
# from models.klue_bert import MyBertModel
from preprocessor import bert_tokenizer
from transformers import AutoModel, AutoTokenizer
from preprocessor import load_data
from models.emotion import EmotionClassifier
import re

MODEL_NAME = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
class Recommends:
    def __init__(self):
        self.data = load_data()
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.emotion_model = EmotionClassifier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    
    @staticmethod
    def cos_sim(A, B):
        return dot(A, B) / (norm(A) * norm(B))

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def w_sim(lyrics_sim, emo_sim, lyrics_weight=0.6):    
        agg_sim = lyrics_sim * lyrics_weight + emo_sim * (1 - lyrics_weight)
        return agg_sim.tolist()  

    def predictions(self, lyrics):
        input_ids, attention_mask = bert_tokenizer(self.tokenizer, lyrics)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        vectors = sentence_embeddings.detach().cpu().numpy()
        return vectors

    def recommends(self, song_id):
        songs_info, sim_vectors, emotion_vectors = self.data
        song = get_song_info(song_id)
        song_pub_date = np.round(int(song["public_date"][:4]), -1)
        code_idx = songs_info[songs_info["song_code"] == song["song_id"]].index.tolist()
        korean_lyrics = re.sub("[^가-힣^\s]", "", song["lyrics"])
        if code_idx:
            vector = sim_vectors[code_idx[0]]
            logits = emotion_vectors[code_idx[0]]
        else:
            if "가사 준비중" in korean_lyrics:
                return "가사가 없습니다."
            _, logits = self.emotion_model.classify(korean_lyrics)
            logits_max = torch.tensor(logits, dtype=torch.float64)
            logits_max = torch.sigmoid(logits_max).numpy()
            emotion = self.emotion_model.get_max_n(logits_max, n=3)
            vector = self.predictions(korean_lyrics)[0]
        songs_info["sim_score"] = [self.cos_sim(vector, c) for c in sim_vectors]
        songs_info["emotion_score"] = [self.cos_sim(logits, c) for c in emotion_vectors]
        songs_info["w_score"] = self.w_sim(songs_info["sim_score"], songs_info["emotion_score"])

        songs_info = songs_info[songs_info["song_code"] != int(song["song_id"])]
        songs_info = songs_info[songs_info["public_date_DV"] >= song_pub_date]
        songs_info = songs_info[songs_info["genre"] == song["genre"]]
        songs_info["melon"] = [f"https://www.melon.com/song/lyrics.htm?songId={song_code}" for song_code in songs_info['song_code']]
        songs_info = songs_info.drop("lyrics", axis=1)
        songs_info = songs_info.sort_values(by=["w_score"], ascending=[False])[:10]
        return songs_info.to_dict("records"), emotion