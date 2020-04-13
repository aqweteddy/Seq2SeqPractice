import random
import logging
import os
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
    def __init__(self, tokenizer, model_path,
                 embed_size=256,
                 hidden_size=256,
                 n_layers=1,
                 device='gpu'
                 ):
        self.tokenizer = tokenizer
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.model = Seq2Seq(len(self.tokenizer),
                             hidden_size,
                             embed_size,
                             n_layers=n_layers,
                             )
        
        self.model = self.model.load_state_dict(torch.load(model_path)).to(self.device)
        self.model.eval()
        
    def infer(self, src: List[str], maxlen=512):
        src = [self.encode(s, maxlen) for s in src]
        src = torch.tensor(src) # [B, maxlen]

        preds = self.model.infer(src) # [B, output_maxlen]


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

