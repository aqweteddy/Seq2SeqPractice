import random
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from beam import BeamSearchNode
from queue import PriorityQueue


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self,
                 embedding,
                 dct_size,
                 embed_size,
                 hidden_size,
                 n_layers,
                 dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = embedding
        # self.embedding = nn.Embedding(dct_size, embed_size)
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
    def __init__(self, embedding, embed_size, hidden_size, output_size, n_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # self.embedding = nn.Embedding(int(output_size), embed_size)
        self.embedding = embedding
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
        output, hidden = self.gru(gru_input.contiguous(), hidden.contiguous())
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
                 dropout=0.5,
                 device='cuda',
                 shared_embedding=True,
                 ):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.dct_size = dct_size  # input and output
        self.hidden_size = hidden_size  # gru
        self.embed_size = embed_size  # embedding
        self.n_layers = n_layers  # GRU layers
        self.dropout = dropout

        self.shared_embed = nn.Embedding(
            dct_size, embed_size)  # shared embedding
        self.encoder = Encoder(self.shared_embed, dct_size, embed_size,
                                hidden_size, n_layers, dropout)
        self.decoder = Decoder(self.shared_embed, embed_size, hidden_size,  # bidirectional GRU encoder
                                dct_size, n_layers, dropout)
        # else:
        #     self.encoder = Encoder(nn.Embedding(dct_size, embed_size), dct_size, embed_size,
        #                            hidden_size, n_layers, dropout)
        #     self.decoder = Decoder(nn.Embedding(dct_size, embed_size), embed_size, hidden_size,  # bidirectional GRU encoder
        #                            dct_size, n_layers, dropout)

    def create_mask(self, src):
        mask = ((src != 0) & (src != 3)).permute(1, 0)  # mask 0 unk, 3 pad
        return mask

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

        for i in range(1, target_size):
            output, hidden = self.decoder(inp, hidden, encoder_outputs, mask)
            outputs[:, i - 1, :] = output  # output: [batch_size, i, dct_size]
            teacher_force = random.random() < teacher_ratio
            top1 = output.argmax(1)
            preds[:, i - 1] = top1
            # preds.append(top1.item())
            inp = target[i] if teacher_force else top1
        return outputs, preds

    def infer(self, src, maxlen=30, topk=3):
        """infer

        Arguments:
            src  -- [batch, src_len]
        """
        src = src.T
        batch_size = src.shape[1]
        hidden, encoder_outputs = self.encoder(src)

        # for each sentence
        result = []
        for idx in range(batch_size):
            result.append(self.beam_search(src[:, idx].unsqueeze(
                1), hidden[:, idx, :].unsqueeze(1), encoder_outputs[:, idx, :].unsqueeze(1)))
        return result

    def beam_search(self, src, decoder_hidden, encoder_output):
        # encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)
        topk = 3
        beam_width = 10
        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([1]).to(self.device)
        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == 2 and n.prevNode:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            with torch.no_grad():
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_output, self.create_mask(src))

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(
                    decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)
        return utterances
