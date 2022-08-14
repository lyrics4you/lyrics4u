import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, load_from_disk
from transformers import BertModel, BertTokenizerFast, DistilBertModel
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

DATA_PATH = "KOTE_relabel/"

class loader:
    def __init__(self):
        pass

    
    def dataset(self, DATA_PATH = DATA_PATH):
        return load_from_disk(DATA_PATH)

    
    def BertModel(self, MODEL_PATH): 
        if MODEL_PATH == "kobert":
            return get_pytorch_kobert_model()[0]            
        return BertModel.from_pretrained(MODEL_PATH, return_dict = False)

    
    def tokenizer(self, MODEL_PATH):
        if MODEL_PATH == 'kobert':
            vocab = get_pytorch_kobert_model()[1]
            tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        else: 
            tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
            vocab = nlp.Vocab(tokenizer.vocab)
        return tokenizer, vocab

    
class preprocess:
    def __init__(self, tokenizer = None, vocab = None, configs = None, num_worker = 5):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = configs["max_len"]
        self.batch_size = configs["batch_size"]
        self.num_worker = num_worker
    
    def tokenize(self, ds):
        return _BERTDataset(ds, "text", "class_idx", self.tokenizer, self.vocab, self.max_len, True, False)

    
    def cvt2dataloader(self, tokenized_ds):
        dataloader = torch.utils.data.DataLoader(tokenized_ds, batch_size= self.batch_size, num_workers= self.num_worker)
        return dataloader


class _BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            tokenizer, max_seq_length=max_len, vocab = vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


