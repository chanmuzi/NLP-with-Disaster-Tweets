from base import BaseModel
from transformers import AutoModelForSequenceClassification

class CEModel(BaseModel):
    def __init__(self,model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self,input_ids,attention_mask):
        output = self.model(input_ids,attention_mask)
        logits = output.logits
        return logits