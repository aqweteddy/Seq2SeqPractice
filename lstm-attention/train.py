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
from model import AttnDecoder, Encoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.basicConfig(level=logging.INFO)


class Seq2SeqTrainer:
    def __init__(self, tokenizer,
                 batch_size=64,
                 hidden_size=256,
                 maxlen=512,
                 com_maxlen=30,
                 lr=2e-5,
                 tf_board_dir='./tfboard_log'
                 ):
        # tokenizer
        self.tokenizer = tokenizer

        # model
        self.encoder = Encoder(
            len(tokenizer), hidden_size=hidden_size).to(DEVICE)
        self.decoder = AttnDecoder(hidden_size=hidden_size, output_size=len(
            tokenizer), dropout_p=0.1, max_length=maxlen).to(DEVICE)

        # tfboard & log
        self.writer = SummaryWriter(tf_board_dir)
        self.log = logging.getLogger('Trainer')
        # self.log.setLevel(logging.INFO)

        # parameters
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.maxlen = maxlen
        self.com_maxlen=com_maxlen

        # optimizer & criterion
        self.criterion = nn.NLLLoss()
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        # self.encoder_opt = torch.optim.SGD(self.encoder.parameters(), lr=lr)
        # self.decoder_opt = torch.optim.SGD(self.decoder.parameters(), lr=lr)

    def save_model(self, filename, step=None):
        torch.save(self.encoder.state_dict(),
                   f'{filename}{"step" + str(step) if step else "model"}_encoder.ckpt')
        torch.save(self.decoder.state_dict(),
                   f'{filename}{"step" + str(step) if step else "model"}_decoder.ckpt')

    def train(self, train_loader,
              test_loader=None,
              test_batch_size=2,
              test_step=1000,
              epochs=10,
              teacher_ratio=0.5,
              save_ckpt_step=2,
              path='./ckpt/'):
        global_step = 0

        for epoch in range(epochs):
            tot_loss = 0
            for n, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1} / {epochs}: ')):
                text, com, tag = batch
                # MAXLEN * BATCH_SIZE
                text = text.to(DEVICE)
                com = com.to(DEVICE)
                tag = tag.to(DEVICE)
                text = torch.transpose(text, 1, 0)
                com = torch.transpose(com, 1, 0)
                loss, topis = self.__train_one(text, com, tag, teacher_ratio)
                tot_loss += loss
                self.writer.add_scalar(
                    f'train/loss', loss, global_step=global_step+1)
                self.writer.flush()

                if (global_step + 1) % 100 == 0:
                    self.log.warning(f'topi {topis}')
                if (global_step + 1) % save_ckpt_step == 0:
                    self.save_model(path, global_step)
                global_step += 1

                if test_loader and (global_step+1) % test_step == 0:
                    for batch in test_loader:
                        text, com, tag = batch
                        # MAXLEN * BATCH_SIZE
                        text = text.to(DEVICE).view(-1, test_batch_size)
                        com = com.to(DEVICE).view(-1, test_batch_size)
                        tag = tag.to(DEVICE)
                        ev = self.__eval_one_batch(text, com, tag, test_batch_size)
                        break
                    for idx, s in enumerate(ev):
                        self.writer.add_text(
                            f"sample{idx}", s, global_step=global_step+1)
                    self.writer.flush()

            self.log.warning(f"loss: {tot_loss / n}")
            self.save_model(path, global_step)

    def __train_one(self, text, com, tag, teacher_ratio):
        self.encoder.train()
        self.decoder.train()

        encoder_hidden = torch.zeros(
            1, self.batch_size, self.hidden_size).to(DEVICE)
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()
        loss = 0

        #! Encoder
        encoder_outputs = torch.zeros(
            self.maxlen, self.hidden_size, device=DEVICE)
        for ei in range(self.maxlen):
            output, encoder_hidden = self.encoder(text[ei], encoder_hidden)
            encoder_outputs[ei] = output[0, 0]

        #! Decoder
        use_teacher_forcing = True if random.random() < teacher_ratio else False
        decoder_input = com[0]
        hidden = encoder_hidden
        topis = []

        if use_teacher_forcing:
            for di in range(1, self.com_maxlen):
                # BATCH * 21128
                output, hidden, atten = self.decoder(
                    decoder_input, hidden, encoder_outputs)
                
                loss += self.criterion(output, com[di])
                decoder_input = com[di]  # teacher forcing
        else:
            for di in range(1, self.com_maxlen):
                output, hidden, atten = self.decoder(
                    decoder_input, hidden, encoder_outputs)
                loss += self.criterion(output, com[di])

                topv, topi = output.topk(1)
                topis.append(topi[0].detach().cpu().numpy())
                decoder_input = topi.squeeze().detach()

        #! loss & optimizer
        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()
        return loss.item() / self.com_maxlen, topis

    def __eval_one_batch(self, text, com, tag, test_batch_size=2):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            encoder_hidden = torch.zeros(
                1, test_batch_size, self.hidden_size).to(DEVICE)
            encoder_outputs = torch.zeros(
                self.maxlen, self.hidden_size, device=DEVICE)
            for ei in range(self.maxlen):
                output, encoder_hidden = self.encoder(text[ei], encoder_hidden)
                encoder_outputs[ei] = output[0, 0]

            decoder_input = com[0]

            decoder_hidden = encoder_hidden
            output_sent = [[] for i in range(test_batch_size)]
            for di in range(self.com_maxlen):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                for i, top in enumerate(topi.detach().cpu().numpy()):
                    if top not in self.tokenizer.all_special_ids:
                        output_sent[i].append(top)

            output_sent = [self.tokenizer.decode(sent) for sent in output_sent]
        return output_sent

    def __filter_special_tokens(self, token_list):
        return [token for token in token_list if token not in self.tokenizer.all_special_tokens]


if __name__ == '__main__':
    import pandas as pd
    TRAIN_SET = 30000
    BATCH_SIZE = 64
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
                             batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE, com_maxlen=30, maxlen=MAXLEN, lr=2e-5)

    trainer.train(train_loader=train_loader, test_loader=test_loader, test_step=100,
                  test_batch_size=2, epochs=10, teacher_ratio=0.65, save_ckpt_step=200)

    trainer.writer.close()
