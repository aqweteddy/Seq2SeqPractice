import random
import logging

import torch
import torch.nn as nn
from apex import amp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from dataset import PttDataset
from model import Seq2Seq


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

        # model
        self.model = Seq2Seq(len(self.tokenizer),
                             hidden_size,
                             embed_size,
                             n_layers=n_layers,
                             dropout=dropout).to(DEVICE)

        # tfboard & log
        self.writer = SummaryWriter(tf_board_dir)
        self.log = logging.getLogger('Trainer')
        # self.log.setLevel(logging.INFO)

        # parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        # optimizer & criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        # self.encoder_opt = torch.optim.SGD(self.encoder.parameters(), lr=lr)
        # self.decoder_opt = torch.optim.SGD(self.decoder.parameters(), lr=lr)

    def save_model(self, filename, step=None):
        torch.save(self.model.state_dict(),
                   f'{filename}{"step" + str(step) if step else "model"}.ckpt')

    def train(self,
              train_loader,
              test_loader=None,
              test_batch_size=2,
              test_step=1000,
              epochs=10,
              teacher_ratio=0.5,
              save_ckpt_step=20,
              path='./ckpt/'):
        global_step = 0
        for epoch in range(epochs):
            tot_loss = 0
            pbar = tqdm(train_loader)
            for n, batch in enumerate(pbar):
                pbar.set_description(f'Epoch {epoch+1} / {epochs}: ')
                text, com, tag = batch
                # MAXLEN
                text, com, tag = text.to(DEVICE), com.to(
                    DEVICE), tag.to(DEVICE)

                loss, preds = self.__train_one(text, com, tag, teacher_ratio)
                tot_loss += loss
                self.writer.add_scalar(
                    f'train/loss', loss, global_step=global_step+1)
                self.writer.flush()

                # if (global_step + 1) % save_ckpt_step == 0:
                #     self.save_model(path, global_step)
                global_step += 1
            pbar.close()
            self.writer.add_text('TrainOutput',
                                 self.tokenizer.decode(preds[0]),
                                 global_step=global_step+1)
            self.log.warning(f"loss: {tot_loss / n}")
            self.save_model(path, global_step)

    def __train_one(self, text, com, tag, teacher_ratio):
        self.model.train()
        outputs, preds = self.model(text, com, tag, teacher_ratio)

        # print(outputs.size(), com.size())
        loss = self.criterion(outputs.transpose(1, 2), com)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), preds

    def __filter_special_tokens(self, token_list):
        return [token for token in token_list if token not in self.tokenizer.all_special_tokens]


if __name__ == '__main__':
    import pandas as pd
    TRAIN_SET = 30000
    BATCH_SIZE = 64
    EMBED_SIZE = 512
    HIDDEN_SIZE = 256
    MAXLEN = 512
    # print("loading data...")
    # df = pd.read_csv('../fetch_data/ptt.csv')
    # print(f"Total: {len(df)}\nTrain: {TRAIN_SET}")
    # df = df.sample(TRAIN_SET)
    # ds = PttDataset(df['text'].to_list(), df['comment'].to_list(), df['tags'].to_list(), maxlen=MAXLEN, com_maxlen=30)
    # ds.save('train_30000.json')

    ds = PttDataset(from_file='./train_30000.json')
    # ds = PttDataset(['sdsss', 'asaaa'], ['aaa', 'b'], [1, 0])
    test_ds = PttDataset(from_file='./test_sample.json')
    train_loader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, shuffle=True, batch_size=2)
    print(f"dct size: {ds.get_dct_size()}")

    trainer = Seq2SeqTrainer(BertTokenizer.from_pretrained('../model/roberta'),
                             embed_size=EMBED_SIZE,
                             hidden_size=HIDDEN_SIZE,
                             n_layers=3,
                             lr=2e-5,
                             dropout=0.5
                             )

    trainer.train(train_loader=train_loader,
                  test_loader=None,
                  test_step=100,
                  test_batch_size=2,
                  epochs=100,
                  teacher_ratio=0.5,
                  save_ckpt_step=200)

    trainer.writer.close()
