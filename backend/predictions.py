import torch
from models.klue_bert import MyBertModel


def predictions(input_ids, attention_mask):
    model = MyBertModel()
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    vectors = outputs.detach().cpu().numpy()
    return vectors
