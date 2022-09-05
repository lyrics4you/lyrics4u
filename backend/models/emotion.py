import os
import pytorch_lightning as pl
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch

LABELS = ['불평/불만',
     '환영/호의',
     '감동/감탄',
     '지긋지긋',
     '고마움',
     '슬픔',
     '화남/분노',
     '존경',
     '기대감',
     '우쭐댐/무시함',
     '안타까움/실망',
     '비장함',
     '의심/불신',
     '뿌듯함',
     '편안/쾌적',
     '신기함/관심',
     '아껴주는',
     '부끄러움',
     '공포/무서움',
     '절망',
     '한심함',
     '역겨움/징그러움',
     '짜증',
     '어이없음',
     '없음',
     '패배/자기혐오',
     '귀찮음',
     '힘듦/지침',
     '즐거움/신남',
     '깨달음',
     '죄책감',
     '증오/혐오',
     '흐뭇함(귀여움/예쁨)',
     '당황/난처',
     '경악',
     '부담/안_내킴',
     '서러움',
     '재미없음',
     '불쌍함/연민',
     '놀람',
     '행복',
     '불안/걱정',
     '기쁨',
     '안심/신뢰']

label_dict = dict(zip(range(len(LABELS)), LABELS))

class EmotionClassifier:
    def __init__(self, label_dict=label_dict, W_PATH = "data/kote_pytorch_lightning.bin"):
        self.model = load_model()
        self.model.load_state_dict(torch.load(W_PATH))
        self.label_dict = label_dict

    def classify(self, text):
        probs, logits = self.model(text)
        return probs[0].detach().cpu().numpy(), logits[0].detach().cpu().numpy()        

    
    def get_max_n(self, values, n = 3):
        max_n_idx = (-values).argsort()[:n]
        max_n_labels, max_n_values = [], []
        values = values.tolist()
        for idx in max_n_idx:
            max_n_values.append(values[idx])
            max_n_labels.append(self.label_dict[idx])
        return max_n_labels, max_n_values
    


class load_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base")
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.classifier = nn.Linear(self.electra.config.hidden_size, 44)
        
    def forward(self, text:str):
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=512,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,   
          return_attention_mask=True,
          return_tensors='pt',
        )
        output = self.electra(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        output = output.last_hidden_state[:,0,:]
        logits = self.classifier(output)
        probs = torch.sigmoid(logits)  
        return probs, logits
