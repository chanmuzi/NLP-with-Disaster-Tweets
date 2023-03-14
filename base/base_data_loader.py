from torch.utils.data import DataLoader,random_split
from .base_dataset import BaseDataset

def BaseDataLoader(train_dataset,test_dataset,tokenizer,batch_size):
    # BaseDataset(df,tokenizer,label,preprocess)
    train_dataset = BaseDataset(train_dataset,tokenizer,label=True,preprocess=False)
    test_dataset = BaseDataset(test_dataset,tokenizer,label=False,preprocess=False)

    train_size = int(len(train_dataset)*0.8)
    valid_size = len(train_dataset) - train_size

    train_dataset,valid_dataset = random_split(train_dataset,[train_size,valid_size])

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,pin_memory=False)

    print(f'{len(train_dataset)} train samples')
    print(f'{len(valid_dataset)} valid samples')
    print(f'{len(test_dataset)} valid samples')

    return train_loader,valid_loader,test_loader