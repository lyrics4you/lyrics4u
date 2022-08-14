import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast


from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from transformers import get_cosine_schedule_with_warmup
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import datetime
import random
import time


DATA_IN_PATH = "./data"
FILE_NAME = "KOTE_relabel.tsv"
DATA_PATH = os.path.join(DATA_IN_PATH, FILE_NAME)
kote = pd.read_csv(DATA_PATH, sep="\t", index_col=0)
# print(kote.shape)
# kote.head()

class KoteDataset(Dataset):

  # 생성자, 데이터를 전처리 하는 부분
  def __init__(self, inputs, targets, tokenizer, max_len):
    self.inputs = inputs
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    # self._prepare_data()

  # def _prepare_data(self):
  #   kote = pd.read_csv(DATA_PATH, sep="\t")
    
  def __len__(self):
    return len(self.inputs)

  # idx(인덱스)에 해당하는 입출력 데이터를 반환
  def __getitem__(self, idx):
    input = str(self.inputs[idx])
    target = self.targets[idx]
    encoding = self.tokenizer.encode_plus(input, 
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                          return_token_type_ids=False, 
                                          padding='max_length', 
                                          truncation=True)
    return {
      'input_text' : input,
      'input_ids' : encoding['input_ids'].flatten(),
      'attention_mask' : encoding['attention_mask'].flatten(),
      'targets' : torch.tensor(target, dtype=torch.long)
    }


def KoteDataLoader(df, tokenizer, max_len, batch_size):
  ds = KoteDataset(
          inputs=df['text'].to_numpy(),
          targets=df['class'].to_numpy(),
          tokenizer=tokenizer,
          max_len=max_len)
  
  return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
  )


def accuracy_measure(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # hh:mm:ss 형태로 변경



def main():
    parser = argparse.ArgumentParser(description="BERT training for sentiment classification for 2 classes")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="pre-trained model name or path")
    parser.add_argument("--seed", type=int, default=42 , help="random seed value")
    parser.add_argument("--batch_size", type=int, default=16,  help="batch size for training")
    parser.add_argument("--max_len", type=int, default=320, help="maximum length of sequence")
    parser.add_argument("--epoch", type=int, default=1, help="number of training epochs")  # 넘 오래걸려서ㅠ 일단은 1로. 나중에 수정
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    args = parser.parse_args()

    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    MODEL_NAME = args.model_name  # 'klue/roberta-base', 'klue/bert-base', 'monologg/koelectra-base-v3-discriminator', 'bert-base-multilingual-cased', ...
    MAX_LEN = args.max_len  # maximum token lenngth
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epoch
    EPS = 1e-8  # optimizer에서 0 나누기 방지를 위한 epsilon

    # Load data
    DATA_DIR = './data'
    FILE_NAME = 'KOTE_relabel.tsv'
    DATA_PATH = os.path.join(DATA_DIR, FILE_NAME)
    kote = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
    train_df = kote[kote['datset']=='train']
    val_df = kote[kote['datset']=='valid']
    test_df = kote[kote['datset']=='test']

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

    train_dataloader = KoteDataLoader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    val_dataloader = KoteDataLoader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)
    # test_dataloader = KorSongsDataLoader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                          num_labels=5,
                                                          output_hidden_states=False,
                                                          output_attentions=False)

    if device.type != 'cpu':
        print("Running model in CUDA")
        model.cuda()

    optimizer = optim.AdamW(model.parameters(),lr=LEARNING_RATE, eps=EPS)
    total_steps = len(train_dataloader) * EPOCHS

    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
    training_stats = []
    scaler = GradScaler()
    total_t0 = time.time()

    for epoch in range(EPOCHS):
        print('')
        print(f'======== Epoch {epoch+1:}/{EPOCHS:} ========')
        print('Training...')

        t0 = time.time()  # 시작 시간 설정
        total_train_loss = 0  # loss 초기화
        
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            if step%250 == 0 and not step==0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step:>5,} of {len(train_dataloader):>5,}.    Elapsed: {elapsed}.')

            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            model.zero_grad()    # 그래디언트 초기화

            '''forward'''
            with autocast():
                loss, logits = model(b_input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=b_input_mask, 
                                     return_dict=False, 
                                     labels=b_labels)
            total_train_loss += loss.item()

            '''backpropagation'''
            scaler.scale(loss).backward()  # 그래디언트 계산
            scaler.step(optimizer)
            scaler.update()  # scaler 업데이트
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
            optimizer.step()  # 그래디언트를 통해 가중치 파라미터 업데이트
            scheduler.step()  # learning rate 업데이트

        avg_train_loss = total_train_loss / len(train_dataloader)  # 평균 loss
        training_time = format_time(time.time() - t0)
        print('')
        print(f'  Average training loss: {avg_train_loss:.2f}')
        print(f'  Training epcoh took: {training_time:}')

        # ----------------------------------------------------------------

        print('')
        print('Running Validation...')

        t0 = time.time()

        model.eval()  # 평가 모드

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            with torch.no_grad():    
                loss, logits = model(b_input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=b_input_mask, 
                                     labels=b_labels)
                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += accuracy_measure(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        
    avg_val_loss = total_eval_loss / len(val_dataloader)

    validation_time = format_time(time.time() - t0)

    print(f"  Validation Loss: {avg_val_loss:.2f}")
    print(f"  Validation took: {validation_time:}")

    training_stats.append(
        {
            'epoch': epoch+1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


    # Save the model
    MODEL_DIR = './model/'
    MODEL_SAVE_NAME = f'kote-trained-by-{MODEL_NAME}'
    MODEL_SAVE_PATH = MODEL_DIR + MODEL_SAVE_NAME
    
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    print(f'saving model to {MODEL_SAVE_PATH}')

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()