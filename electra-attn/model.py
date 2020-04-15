import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel

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
                          bidirectional=True)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input):
        # input = [batch, seq]
        embedded = self.embedding(input)
        self.gru.flatten_parameters()

        output, hidden = self.gru(self.dropout(embedded))
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_size)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # print(hidden.size())
        hidden = torch.tanh(self.fc(hidden))
        # print(hidden.size())
        return hidden, output


class Attention(nn.Module):
    def __init__(self, enc_hid_size, dec_hid_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_size * 2 + dec_hid_size, dec_hid_size)
        self.v = nn.Linear(dec_hid_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        """forward

        Arguments:
            hidden {[type]} -- [n_layer, batch size, hidden_size]
            encoder_outputs {[type]} -- [src_len, batch_size, hidden_size * 2]
        """
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # hidden = [batch_size, src_len, hidden_size]
        hidden = hidden[-1, :, :]  # last layer
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs = [batch_size, src_len, hid_size * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # energy = [bach_size, src_len, hidden_size]
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(int(output_size), embed_size)
        self.attention = Attention(hidden_size, hidden_size)

        self.gru = nn.GRU(embed_size + hidden_size * 2, hidden_size, n_layers)
        self.fc1 = nn.Linear(hidden_size * 2 + hidden_size +
                             embed_size, output_size)
        # self.fc2 = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        """forward

        Arguments:
            input {} -- [batch_size]
            hidden {} -- [batch_size, hidden_size]

        Keyword Arguments:
            encoder_outputs  -- [src_len, batch size, hidden_size * 2] (default: {None})

        """
        input = input.unsqueeze(0)  # input: [1, batch_size]
        embed = self.embedding(input)
        embed = self.dropout(embed)  # [1, batch_size, emb_size]

        # attn: [batch_size, src_len]
        attn = self.attention(hidden, encoder_outputs, mask)
        attn = attn.unsqueeze(1)  # [1, batch_size, src_len]
        # encoder_outputs: [batch_size, src_len, hidden_size * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(attn, encoder_outputs).permute(
            1, 0, 2)  # weighted: [1, batch_size, hidden_size * 2]

        # gru_input: [1, batch_size, hidden_size * 2 + embed_size]
        gru_input = torch.cat((embed, weighted), dim=2)

        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)
        # output: [1, batch_size, hidden_size]
        # hidden: [n_layer, batch_size, hidden_size]

        output = output.squeeze(0)
        embed = embed.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.fc1(torch.cat((output, embed, weighted), dim=1))
        # output = self.fc2(output)  # batch_size, output_size
        return output, hidden  # hidden: [n_layer, batch_size, hidden_size]


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
        self.hidden_size = hidden_size  # gru
        self.embed_size = embed_size  # embedding
        self.n_layers = n_layers  # GRU layers
        self.dropout = dropout
        self.encoder = Encoder(dct_size, embed_size,
                               hidden_size, n_layers, dropout)
        self.decoder = Decoder(embed_size, hidden_size,  # bidirectional GRU encoder
                               dct_size, n_layers, dropout)

    def create_mask(self, src):
        return (src != 0).permute(1, 0)  # mask 0 padding

    def forward(self, src, target, teacher_ratio=0.5):
        src, target = src.T, target.T
        batch_size = src.shape[1]
        target_size = target.shape[0]
        # encoder
        # encoder_outputs: batch_size, seq_len, hidden_size * n_direction
        # hidden: n_layer * n_direction, batch_size, hidden_size
        hidden, encoder_outputs = self.encoder(src)
        # decoder
        mask = self.create_mask(src)
        outputs = torch.zeros(batch_size, target_size, self.dct_size).cuda()
        preds = torch.zeros(batch_size, target_size).cuda()
        inp = target[0, :]

        self.inp_backup = inp

        for i in range(0, target_size-1):
            output, hidden = self.decoder(inp, hidden, encoder_outputs, mask)
            outputs[:, i, :] = output  # output: [batch_size, i, dct_size]
            teacher_force = random.random() < teacher_ratio
            top1 = output.argmax(1)
            preds[:, i] = top1
            # preds.append(top1.item())
            inp = target[i+1] if teacher_force else top1
        return outputs, preds

    def infer(self, src, topk=3):
        """infer

        Arguments:
            src  -- [batch, src_len]
        """

    
    def beam_search(src, topk):
        pass
