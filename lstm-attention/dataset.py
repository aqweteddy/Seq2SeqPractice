from torch.utils.data.dataset import Dataset
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import json

class PttDataset(Dataset):
    def __init__(self, texts=None, comments=None, tags=None, from_file=None, maxlen=512, com_maxlen=30, train=True):
        self.tokenizer = BertTokenizer.from_pretrained('../model/roberta')

        if from_file:
            with open(from_file, 'r', encoding='utf-8') as f:
                total = json.load(f)
                self.texts = total['texts']
                self.comments = total['comments']
                self.tags = total['tags']
            print(f"data size: {len(self.texts)}")
            assert len(self.texts) == len(self.comments)
            assert len(self.comments) == len(self.tags)
            self.train = True

            return


        self.texts = []
        self.comments = []
        self.tags = []
        self.train = True
        self.maxlen = maxlen

        for text, comment, tag in zip(tqdm(texts, desc='Encoding...'), comments, tags):
            try:
                text = self.encode(text, maxlen)
                comment = self.encode(comment, com_maxlen)
                self.texts.append(text)
                self.comments.append(comment)
                self.tags.append(tag)
            except:
                pass
        
        assert len(self.texts) == len(self.comments)
        assert len(self.comments) == len(self.tags)
    
    def encode(self, text, maxlen):
        inp_ids = self.tokenizer.encode(text, max_length=maxlen)
        pad_len = maxlen - len(inp_ids)
        return  inp_ids + ([0] * pad_len)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({'texts': self.texts, 'comments': self.comments, 'tags': self.tags}, f)

    def get_dct_size(self):
        return len(self.tokenizer)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, k):
        if self.train:
            # print(f"""
            # {torch.tensor(self.texts[k]).size()}
            # {torch.tensor(self.comments[k]).size()}
            # {torch.tensor(self.comments[k]).size()}
            # """)
            return (torch.tensor(self.texts[k]), 
            torch.tensor(self.comments[k]), 
            torch.tensor(self.tags[k]))
        else:
            return torch.tensor(self.texts[k])


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../fetch_data/ptt.csv')
    df = df.head(5)
    ds = PttDataset(df['text'].to_list(), df['comment'].to_list(), df['tags'].to_list())
    print(ds[0])