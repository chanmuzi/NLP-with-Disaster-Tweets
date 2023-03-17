from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss

def cross_entropy_loss(preds, labels):
    return CrossEntropyLoss()(preds, labels)

def binary_cross_entropy_loss(preds, labels):
    return BCELoss()(preds, labels)