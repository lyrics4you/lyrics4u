from transformers import BertModel
import torch.nn as nn


class BertModel(nn.Module):
    def __init__(self, model_name="klue/bert-base"):
        super(BertModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask)["pooler_output"]
