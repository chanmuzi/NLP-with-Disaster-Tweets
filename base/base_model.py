import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model

    def forward(self,input_ids,attention_mask):
        """
        When you use original model, you don't have to make logit
        be picked out from the output.
        But, in case of 'model for classification', the output of
        model includes 'loss' and 'logits'.
        """
        output = self.model(input_ids,attention_mask)
        logits = output.logits
        return logits