import random
import logging
import os
import json
from typing import List

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from vocab import Vocab
from dataset import PttDataset
from model import Seq2Seq
from transformers import BertTokenizer


class Inference:
    def __init__(self, tokenizer, model_path, config_file=None,
                 embed_size=256,
                 hidden_size=256,
                 n_layers=3,
                 device='cuda'
                 ):
        self.tokenizer = tokenizer
        self.device = device
        if config_file:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.embed_size = config['embed_size']
                self.hidden_size = config['hidden_size']
                self.n_layers = config['n_layers']
        else:
            self.embed_size = embed_size
            self.hidden_size = hidden_size
            self.n_layers = n_layers

        self.model = Seq2Seq(len(self.tokenizer),
                             hidden_size,
                             embed_size,
                             n_layers=n_layers,
                             device=device
                             )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        # self.model.eval()

    def infer(self, src: List[str], maxlen=512):
        src = [self.encode(s, maxlen) for s in src]
        src = torch.tensor(src).to(self.device) # [B, maxlen]

        preds = self.model.infer(src) # [B, output_maxlen]
        preds = [self.decode(p) for pred in preds for p in pred]
        return preds

    def decode(self, code:List[int]):
        code = [int(c.detach().cpu().numpy()) for c in code]
        result = self.tokenizer.decode(code, remove_special_token=True)
        return result

    def encode(self, doc, maxlen):
        code = self.tokenizer.encode(doc)
        if len(code) > maxlen:
            code = code[:maxlen]
            code[-1] = self.tokenizer.sep_token_id
        elif len(code) < maxlen:
            pad_size = maxlen - len(code)
            code = code + pad_size * [self.tokenizer.pad_token_id]
        assert len(code) == maxlen
        return code

if __name__ == "__main__":
    inf = Inference(tokenizer=Vocab.from_pretrained('./model/vocab.txt'), model_path='model/30.ckpt', device='cuda')
    print(inf.infer(['謝謝各位大大幫忙高調與置頂 目前已找到影像，稍晚會進行刪文感謝各位', 
                    '徵求說明：朋友為開計程車 與今早上與一小孩造成交通事故，小孩當時手發生骨折，目前缺少影像釐清雙方責任，麻煩各位幫忙協詢，謝謝']))
