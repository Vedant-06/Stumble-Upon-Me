from transformers import AutoModel
import config
import transformers
import torch.nn as nn

class XLMRBase(nn.Module):
    def __init__(self):
        super(XLMRBase, self).__init__()
        self.XLMR = AutoModel.from_pretrained("xlm-roberta-base")
        self.XLMR_drop = nn.Dropout(0.3)
        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1,768))
        self.Dense1 = nn.Linear(768,128)
        self.Dense2 = nn.Linear(128,64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(64,1)

    def forward(self, ids, mask, token_type_ids):
        o1 = self.XLMR(ids, attention_mask=mask, token_type_ids=token_type_ids)[0]
        bo = self.XLMR_drop(o1)
        bo = self.AvgPool(bo).squeeze(1)
        bo = self.relu(self.Dense1(bo))
        bo = self.relu(self.Dense2(bo))
        bo = self.out(bo)
        output = self.sigmoid(bo)
        return output
