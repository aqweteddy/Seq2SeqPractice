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
                 n_layers=2,
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
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.to(self.device)
        self.model.eval()

        # self.model.eval()

    def infer(self, src: List[str], maxlen=512):
        src = [self.encode(s, maxlen) for s in src]
        src = torch.tensor(src).to(self.device)  # [B, maxlen]

        preds = self.model.infer(src)  # [B, output_maxlen]
        preds = [self.decode(p) for pred in preds for p in pred]
        return preds

    def decode(self, code: List[int]):
        code = [int(c.detach().cpu().numpy()) for c in code]
        result = self.tokenizer.decode(code, remove_special_token=False)
        return result

    def encode(self, doc, maxlen):
        code = self.tokenizer.encode(doc)
        if len(code) > maxlen:
            code = code[:maxlen]
            code[-1] = self.tokenizer.eos_id
        elif len(code) < maxlen:
            pad_size = maxlen - len(code)
            code = code + pad_size * [self.tokenizer.pad_token_id]
        assert len(code) == maxlen
        return code


if __name__ == "__main__":

    # from pprint import pprint
    inf = Inference(tokenizer=Vocab.from_pretrained('./model/vocab.txt'),
                    model_path='model/28.ckpt', 
                    config_file='model/config.json', 
                    device='cpu')
    print(inf.infer(['國際中心／綜合報導英國薩福克郡32歲的巴萊塔(Becky[UNK]Barletta)在新婚不久後，被診斷出罹患失智症，是英國最年輕的失智症患者之一，壽命可能只剩5年。巴萊塔的父親難過的表示，據《每日郵報》報導，巴萊塔目前住在自己的娘家，因為她已經無法自理生活，需要家人全天候的照顧。巴萊塔2015年10月結婚，但在2016年性情大變，當年8月確診罹患「上額顳葉失智症」(Frontotemporal[UNK]dementia)。巴萊塔罹病後，外在的行為表現、情緒、社交及語言能力都受影響。事實上，巴萊塔原是一名滑雪教練，學生們都非常喜歡她，怎料結婚後突然改變，讓家人不能接受。據了解，巴萊塔的叔叔及母親的表弟都死於失智症，因此家人非常擔心她的狀況。巴萊塔的妹妹蘇菲(Sophie)難過的表示，其實姊姊在結婚前，，「她以前是個很棒的老師，尤其對孩子特別好，大家都很喜歡她。」專家表示，若是巴萊塔的病情持續惡化，未來就連吃飯、說話都會有問題，甚至活不過10年。蘇菲補充，「姊姊會突然向街上的人說話，問他們能不能發出些好笑的聲音。大家都不明白為什麼她的外表看起來如此正常，卻會對人如此沒禮貌。」蘇菲目前正在向各方發起募款活動，希望能夠讓外界更']))
