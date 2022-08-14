import torch
from torch import nn
import torch.nn.functional as F
from transformers import AdamW, TrainingArguments
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm, tqdm_notebook
import numpy as np
from datasets import Dataset

class _BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes= 5,   ##클래스 수 조정##
                 dr_rate=0.5,
                 params=None):
        super(_BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)



class engine:
    def __init__(self, bertmodel, device, configs):
#         super(modelEngine, self).__init__()
        self.num_classes = configs["num_classes"]
        self.device = device
        self.model = _BERTClassifier(bertmodel, num_classes = self.num_classes).to(self.device)
        self.optimizer = None
        self.loss_fn = None
        self.t_total = None
        self.warmup_step = None
        self.scheduler = None
        self.metric = None
        
        
    def set_default(self, configs):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=configs["learning_rate"])
        self.loss_fn = nn.CrossEntropyLoss()
        self.t_total = configs["train_len"] * configs["num_epochs"]
        self.warmup_step = int(self.t_total * configs["warmup_ratio"])
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps= self.warmup_step, num_training_steps= self.t_total)
        self.metric = calc_accuracy
        
        
    def fit(self, trainset = None, validset = None, configs = None):
        for e in range(configs["num_epochs"]):
            train_acc = 0.0
            test_acc = 0.0
            self.model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(trainset)):
                self.optimizer.zero_grad()
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                label = label.long().to(self.device)
                out = self.model(token_ids, valid_length, segment_ids)
                loss = self.loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), configs["max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                train_acc += calc_accuracy(out, label)
                if batch_id % configs["log_interval"] == 0:
                    print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
            if validset:
                self.model.eval()
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(validset)):
                    token_ids = token_ids.long().to(self.device)
                    segment_ids = segment_ids.long().to(self.device)
                    label = label.long().to(self.device)
                    with torch.no_grad():
                        out = self.model(token_ids, valid_length, segment_ids)
                    test_acc += calc_accuracy(out, label)
                print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))  

                
    def predict(self, new_text:list, preprocessor, return_max_only = True):
        data_len = len(new_text)
        new_ds = Dataset.from_dict({"text" : new_text, "class_idx": [0] * data_len})
        tok_new_ds = preprocessor.tokenize(new_ds)
        dataloader = preprocessor.cvt2dataloader(tok_new_ds)
        probs_array = []
        self.model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            label = label.long().to(self.device)
            with torch.no_grad():
                out = self.model(token_ids, valid_length, segment_ids)                
#             logits = np.array([logit.detach().cpu().numpy() for logit in out])
            probs = F.softmax(out, dim = 1).detach().cpu().numpy()
            probs_array.extend(probs)
        probs_array = np.array(probs_array)
        if return_max_only:
            return (np.argmax(probs_array, axis = 1), probs_array)
        else:
            return ((-probs_array).argsort(axis = 1), probs_array)

        
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

  


        
    
    

