import random
import logging
import json
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from vocab import Vocab
from dataset import PttDataset
from model import Seq2Seq
from transformers import BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.basicConfig(level=logging.INFO)


class Seq2SeqTrainer:
    def __init__(self, tokenizer,
                 embed_size=256,
                 hidden_size=256,
                 n_layers=1,
                 lr=2e-5,
                 dropout=0.5,
                 tf_board_dir='./tfboard_log'
                 ):
        # tokenizer
        self.tokenizer = tokenizer

        # Arguments
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lr = lr
        self.dropout = dropout

        # model
        self.model = Seq2Seq(len(self.tokenizer),
                             hidden_size,
                             embed_size,
                             n_layers=n_layers,
                             dropout=dropout).to(DEVICE)
        self.model.apply(self.init_weights)
        self.model = nn.DataParallel(self.model)
        # tfboard & log
        self.writer = SummaryWriter(tf_board_dir)
        self.log = logging.getLogger('Trainer')
        # self.log.setLevel(logging.INFO)
        self.log.warning(f'CUDA count: {torch.cuda.device_count()}')

        # parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        # optimizer & criterion
        parameters_num = sum(p.numel()
                             for p in self.model.parameters() if p.requires_grad)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.log.warning(f'trainable parameters: {parameters_num}')

    def train(self,
              train_loader,
              test_loader=None,
              test_batch_size=2,
              test_step=1000,
              epochs=10,
              teacher_ratio=0.5,
              save_ckpt_step=20,
              path='./model/'):
        global_step = 0
        self.save_config(os.path.join(path, 'config.json'))

        for epoch in range(epochs):
            tot_loss = 0
            pbar = tqdm(train_loader)
            for n, batch in enumerate(pbar):
                pbar.set_description(f'Epoch {epoch+1} / {epochs}: ')
                text, title = batch
                text, title = text.cuda(), title.cuda()
                loss, preds = self.__train_one(text, title, teacher_ratio)

                tot_loss += loss
                self.writer.add_scalar(
                    f'train/loss', loss, global_step=global_step+1)
                self.writer.flush()

                if (global_step) % 200 == 0:
                    text = text.detach().cpu().tolist()
                    title = title.detach().cpu().tolist()
                    preds = preds.detach().cpu().tolist()
                    output = f"title:{''.join(self.tokenizer.decode(title[2]))}\ntext: {''.join(self.tokenizer.decode(text[2]))}\npred:{''.join(self.tokenizer.decode(preds[2]))} "
                    self.writer.add_text('TrainOutput',
                                         output,
                                         global_step=global_step+1)

                # if (global_step + 1) % save_ckpt_step == 0:
                #     self.save_model(path, global_step)
                global_step += 1
            pbar.close()

            self.log.warning(f"loss: {tot_loss / n}")
            self.save_model(f'{os.path.join(path, str(epoch+1))}.ckpt')

    def save_model(self, filename):
        torch.save(self.model.module.state_dict(),
                   f'{filename}')

    def save_config(self, filename):
        with open(filename, 'w') as f:
            json.dump(dict(embed_size=self.embed_size,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           lr=self.lr,
                           dropout=self.dropout
                           ), f)

    @staticmethod
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def __train_one(self, text, title, teacher_ratio):
        self.model.train()
        # output: batch_size, title_len, dic_len
        outputs, preds = self.model(text, title, teacher_ratio)
        outputs = outputs.reshape(-1, len(self.tokenizer))
        title = title.reshape(-1)
        # print(outputs.shape, title.shape)
        loss = self.criterion(outputs, title).mean()

        # self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        return loss.item(), preds

    def evaluate(self, loader):
        self.model.eval()
        loss = 0

        for i, batch in enumerate(tqdm(loader)):
            text, title = batch
            outputs, preds = self.model(text, title, 0)
            # self.log.warning(preds.shape)
            # outputs: target_size - 1, batch_size, dct_size
            outputs = outputs[1:].view(-1, len(self.tokenizer))
            title = title[1:].view(-1)  # [(target_size-1) * batch_size]
            loss += criterion(outputs, title).item()

        return loss / len(loader)

    def __filter_special_tokens(self, token_list):
        return [token for token in token_list if token not in self.tokenizer.all_special_tokens]


if __name__ == '__main__':
    import pandas as pd
    from vocab import Vocab

    BATCH_SIZE = 32 * 8
    EMBED_SIZE = 256
    HIDDEN_SIZE = 256

    ds = PttDataset(from_file='ettoday_char.json')

    # test_ds = PttDataset(from_file='./test.json')
    train_loader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)
    # test_loader = DataLoader(test_ds, shuffle=True, batch_size=2)
    print(f"dct size: {ds.get_dct_size()}")

    trainer = Seq2SeqTrainer(Vocab.from_pretrained('./model/vocab.txt'),
                             embed_size=EMBED_SIZE,
                             hidden_size=HIDDEN_SIZE,
                             n_layers=3,
                             lr=1e-4,
                             dropout=0.5
                             )

    trainer.train(train_loader=train_loader,
                  test_loader=None,
                  test_step=100,
                  test_batch_size=2,
                  epochs=40,
                  teacher_ratio=0.5, 
                  save_ckpt_step=200)

    trainer.writer.close()
