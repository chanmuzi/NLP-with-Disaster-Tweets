import torch.nn as nn
import numpy as np
from abc import abstractmethod
# abstract base class

class BaseModel(nn.Module):

    @ abstractmethod
    def forward(self,*inputs):
        """
        When you use original model, you don't have to make logit
        be picked out from the output.
        But, in case of 'model for classification', the output of
        model includes 'loss' and 'logits'.
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)