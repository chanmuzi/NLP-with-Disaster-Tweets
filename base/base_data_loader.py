from torch.utils.data import DataLoader,random_split
from .base_dataset import BaseDataset

def BaseDataLoader(dataset,tokenizer,batch_size,is_test):
    # BaseDataset(df,tokenizer,label,preprocess)
    if is_test == True:
        test_dataset = BaseDataset(dataset, tokenizer, label=False, preprocess=False)
        test_loader = DataLoader(test_dataset, batchsize=batch_size, shuffle=False, pin_memory=False)

        print(f'{len(test_dataset)} valid samples')

        return test_loader
    else:
        train_dataset = BaseDataset(dataset,tokenizer,label=True,preprocess=False)

        train_size = int(len(train_dataset)*0.8)
        valid_size = len(train_dataset) - train_size

        train_dataset,valid_dataset = random_split(train_dataset,[train_size,valid_size])

        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
        valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

        print(f'{len(train_dataset)} train samples')
        print(f'{len(valid_dataset)} valid samples')
    
        return train_loader,valid_loader,