import torch.nn as nn
import torch
import torch.nn.functional as F

import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self,
                 dct_size,
                 embed_size,
                 hidden_size,
                 n_layers,
                 dropout=0.5):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(dct_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, batch_first=True,  bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch, seq]
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)

        return output, hidden


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(int(output_size), embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.fc3 = nn.Linear(hidden_size * 4, output_size)

    def forward(self, input, hidden, encoder_outputs=None):
        # TODO encoder_outputs if attn
        embed = self.embedding(input)
        embed = self.dropout(embed)
        output, hidden = self.gru(embed, hidden)
        output = self.fc1(output.squeeze(1))
        output = self.fc2(output)
        output = self.fc3(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self,
                 dct_size,
                 hidden_size,
                 embed_size,
                 n_layers=1,
                 dropout=0.5
                 ):
        super(Seq2Seq, self).__init__()
        self.dct_size = dct_size  # input and output
        self.hidden_size = hidden_size # gru
        self.embed_size = embed_size  # embedding
        self.n_layers = n_layers  # GRU layers
        self.dropout = dropout
        self.encoder = Encoder(dct_size, embed_size,
                               hidden_size, n_layers, dropout)
        self.decoder = Decoder(embed_size, hidden_size * 2, # bidirectional GRU encoder
                               dct_size, n_layers, dropout)

    def forward(self, inp, target, tag, teacher_ratio=0.5):
        """
        inp: BATCH * MAX_LEN
        target: BATCH * MAXLEN
        """
        batch_size = inp.shape[0]
        target_size = target.shape[1]
        # encoder
        encoder_outputs, hidden = self.encoder(inp)
        # encoder_outputs: batch_size, seq_len, hidden_size * n_direction
        # hidden: n_layer * n_direction, batch_size, hidden_size

        # concat bidirectional
        hidden = hidden.view(self.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        # decoder
        target = target.unsqueeze(2)  # BATCH * MAXLEN * 1
        inp = target[:, 0, :]  # 101 (start token) BATCH_SIZE * 1 * 1
        outputs = torch.zeros(batch_size, target_size,
                              self.dct_size).to(DEVICE)
        outputs[:, 0, :] = inp
        preds = []
        for i in range(1, target_size):
            output, hidden = self.decoder(inp, hidden, encoder_outputs)
            outputs[:, i, :] = output  # batch insert No.i chars [BATCH, 21128]
            top1 = output.argmax(1).unsqueeze(1)
            teacher_fl = random.random() < teacher_ratio
            inp = target[:, i, :] if teacher_fl else top1
            preds.append(top1)
        preds = torch.cat(preds, 1)
        return outputs, preds
