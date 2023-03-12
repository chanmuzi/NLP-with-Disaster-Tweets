import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self,df,tokenizer,label):
        self.df = df
        self.tokenizer = tokenizer
        self.label

    def __getitem__(self,idx):
        text = self.df.loc[idx]['text']

        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=82,
            return_tensors='pt',
            return_attention_mask=True,
        )

        if self.label:
            labels = self.df.loc[idx]['target']
            return {'input_ids':encoded_dict['input_ids'].squeeze(),
                    'attention_mask':encoded_dict['attention_mask'].squeeze(),
                    'labels':torch.tensor(encoded_dict['lables'],dtype=torch.long).unsqueeze(dim=0)}
        else:
            return {'input_ids':encoded_dict['input_ids'].squeeze(),
                    'attention_mask':encoded_dict['attention_mask'].squeeze()}
                    
    def __len__(self):
        return len(self.df)