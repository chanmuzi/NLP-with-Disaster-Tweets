from sklearn.metrics import f1_score

def accuracy(y_true, y_pred):
    return f1_score(y_true, y_pred)