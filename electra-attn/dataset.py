from torch.utils.data.dataset import Dataset
import torch
# from vocab import Vocab
# from transformers import BertTokenizer
from vocab import Vocab
from tqdm import tqdm
import json, re


class PttDataset(Dataset):
    def __init__(self, docs=None, title=None, from_file=None, maxlen=512, title_maxlen=30):
        # self.tokenizer = BertTokenizer.from_pretrained('../model/roberta/vocab.txt')

        self.tokenizer = Vocab.from_pretrained('./model/vocab.txt')
        if from_file:
            with open(from_file, 'r', encoding='utf-8') as f:
                total = json.load(f)
                self.text = total['text']
                self.title = total['title']
            print(f"data size: {len(self.text)}")
        else:
            self.title = [self.encode_title(doc, maxlen=title_maxlen) for doc in tqdm(title)]
            self.text = [self.encode(doc, maxlen=maxlen) for doc in tqdm(docs)]

        assert len(self.title) == len(self.text)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({'text': self.text, 'title': self.title}, f)

    def get_dct_size(self):
        return len(self.tokenizer)
    
    def encode_title(self, doc, maxlen):
        if ']' in doc:
            doc = ''.join(doc.split(']')[1:])
        return self.encode(doc, maxlen)
    
    def encode(self, doc, maxlen):
        punctuation = "＃＄％＆＇（）◆＊+——！\n，。[]？、~@#￥%……&.+: -*/`~*（＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
        doc = re.sub(f'^[{punctuation}]', ' ', doc)
        code = self.tokenizer.encode(doc)
        if len(code) > maxlen:
            code = code[:maxlen]
            # code[-1] = self.tokenizer.sep_token_id
            code[-1] = self.tokenizer.eos_id
        elif len(code) < maxlen:
            pad_size = maxlen - len(code)
            code = code + pad_size * [self.tokenizer.pad_token_id]
        assert len(code) == maxlen
        return code

    def __len__(self):
        return len(self.text)

    def __getitem__(self, k):
            # print(f"""
            # {torch.tensor(self.texts[k]).size()}
            # {torch.tensor(self.comments[k]).size()}
            # {torch.tensor(self.comments[k]).size()}
            # """)
        return (torch.tensor(self.text[k]), 
        torch.tensor(self.title[k]))

if __name__ == '__main__':
    import pandas as pd

    # df = pd.read_csv('../fetch_data/all.csv')
    df = pd.read_json('../../corpus/ettoday_2017.json')
    df = df.head(100000)
    ds = PttDataset(df['content'].to_list(), df['title'].to_list())
    ds.save('ettoday_char.json')
    # print(ds[0])
    # ds = PttDataset(from_file='gossip.json')
    print(ds[3])