from nltk.corpus import stopwords
from nltk import word_tokenize
import re,string
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self,df,tokenizer,label,preprocess):
        self.df = df
        self.tokenizer = tokenizer
        self.label = label
        self.preprocess = preprocess

    def __getitem__(self,idx):
        text = self.df.loc[idx]['text']

        if self.preprocess:
            text = Preprocessor(text)

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
                    'labels':torch.tensor(labels,dtype=torch.long)}
        else:
            return {'input_ids':encoded_dict['input_ids'].squeeze(),
                    'attention_mask':encoded_dict['attention_mask'].squeeze()}

    def __len__(self):
        return len(self.df)

class Preprocessor():
    def __init__(self,text):
        self.text = text
    
    def remove_tag(self):
        tag = re.compile(r'@\S+')
        self.text = re.sub(tag,'',self.text)

    def remove_URL(self):
        url = re.compile(r'https?:://\S+|www\.\S+')
        self.text = re.sub(url,'',self.text)
    
    def remove_html(self):
        html = re.compile(r'<[^>]+>|\([^)]+\)')
        self.text = re.sub(html,'',self.text)
    
    def remove_punct(self):
        punct = list(string.punctuation)
        tabel = str.maketrans('','',''.join(punct))
        self.text = self.text.translate(tabel)

    def remove_stopwords(self):
        stops = set(stopwords.words('english'))
        words = word_tokenize(self.text)
        self.text = ' '.join([word for word in words if word not in stops])

    def preprocess_all(self):
        self.remove_tag()
        self.remove_URL()
        self.remove_html()
        self.remove_punct()
        self.remove_stopwords()
        return self.text