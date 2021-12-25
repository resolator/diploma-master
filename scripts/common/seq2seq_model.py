#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Seq2Seq model with Bahdanau attention."""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Seq2seqModel(nn.Module):
    def __init__(self,
                 i2c,
                 text_max_len,
                 enc_hs=128,
                 dec_hs=128,
                 emb_size=128,
                 enc_n_layers=1,
                 dec_n_layers=1,
                 dropout_p=0.1,
                 pe=False,
                 teacher_rate=0.9):
        super().__init__()
        self.i2c = i2c
        self.c2i = {c: idx for idx, c in enumerate(i2c)}
        sos_idx = self.c2i['ś']
        alpb_size = len(self.i2c)

        self.fe = FeatureExtractor()
        self.encoder = Encoder(input_sz=256,
                               hs=enc_hs,
                               n_layers=enc_n_layers)
        self.pe = PositionalEncoder(enc_hs * 2) if pe else None
        self.decoder = Decoder(text_max_len=text_max_len,
                               sos_idx=sos_idx,
                               hs=dec_hs,
                               emb_size=emb_size,
                               alphabet_size=alpb_size,
                               dropout_p=dropout_p,
                               n_layers=dec_n_layers,
                               teacher_rate=teacher_rate)
        self.loss_f = nn.NLLLoss(reduction='none', ignore_index=sos_idx)

    def calc_loss(self, logs_probs, targets, targets_lens):
        logs_probs = logs_probs.permute(0, 2, 1)  # BS, AS, W
        loss = self.loss_f(logs_probs, targets)

        # lets mask the loss
        mask = torch.zeros(loss.shape, dtype=torch.int64)

        for i, t_len in enumerate(targets_lens):
            mask[i, :t_len] = 1

        loss = torch.mean(loss[mask])

        return loss# BS, C, W

    def forward(self, x, target_seq=None):
        y = self.fe(x)
        y = self.encoder(y)

        if self.pe is not None:
            y = self.pe(y)

        return self.decoder(y, target_seq)


# dont forget about FPN
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.fe(x).squeeze(2)  # BS, 256, W / 4


class Encoder(nn.Module):
    def __init__(self, input_sz=256, hs=256, n_layers=2):
        super().__init__()
        self.hs = hs
        self.n_layers = n_layers
        self.num_directions = 2  # bidirectional

        self.lstm = nn.LSTM(input_sz,
                            self.hs,
                            num_layers=n_layers,
                            bidirectional=True)

    def forward(self, x):
        # BS, C, W
        x = x.permute(2, 0, 1)  # W, BS, C
        y, _ = self.lstm(x)  # W, BS, 2HS

        # return y.permute(1, 0, 2)  # BS, W, 2HS
        return y.permute(1, 2, 0)  # BS, 2HS, W


class Decoder(nn.Module):
    def __init__(self,
                 text_max_len,
                 sos_idx=0,
                 hs=256,
                 emb_size=256,
                 alphabet_size=81,
                 dropout_p=0.1,
                 n_layers=1,
                 teacher_rate=0.9,
                 pad_size=550):
        super().__init__()
        self.sos_idx = sos_idx
        self.n_layers = n_layers
        self.hs = hs
        self.alphabet_size = alphabet_size
        self.dropout_p = dropout_p
        self.emb_size = emb_size
        self.max_len = text_max_len
        self.teacher_rate = teacher_rate
        self.pad_size = pad_size

        self.emb = nn.Embedding(alphabet_size, self.emb_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.emb_size + self.pad_size,
                            self.hs,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.attention = BahdanauAttention(hs, key_size=self.pad_size)
        self.linear_1 = nn.Linear(self.emb_size + self.pad_size + hs, hs)
        self.linear_2 = nn.Linear(self.hs, self.alphabet_size)

    def forward_step(self, x, enc_out, proj_key, hc):
        query = hc[0][-1].unsqueeze(1)  # BS, 1, HS

        context, attn_probs = self.attention(query, proj_key, enc_out)

        rnn_x = torch.cat([x, context], dim=2)
        y, hc = self.lstm(rnn_x, hc)

        pre_output = torch.cat([x, y, context], dim=2)
        pre_output = self.dropout(pre_output)
        logits = self.linear_1(pre_output)

        return y, hc, logits, attn_probs

    def forward(self, enc_out, target_seq=None):
        if enc_out.size(2) < self.pad_size:
            diff = self.pad_size - enc_out.size(2)
            enc_out = F.pad(enc_out, [0, diff], mode='constant', value=0)
        proj_key = self.attention.key_layer(enc_out)
        
        bs = enc_out.shape[0]
        h, c = self.init_hidden(bs, enc_out.device)
        x = torch.ones(
            (bs, 1),
            dtype=torch.int64,
            device=enc_out.device
        ) * self.sos_idx

        decoder_states = []
        pre_output_vectors = []
        logs_probs = []
        preds = []
        attentions = []
        for i in range(self.max_len):
            prev_embed = self.emb(x)

            output, (h, c), pre_output, attn_probs = self.forward_step(
                prev_embed, enc_out, proj_key, (h, c)
            )
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)
            attentions.append(attn_probs)

            # BS, HS
            log_probs = F.log_softmax(self.linear_2(pre_output[:, -1]), dim=-1)
            logs_probs.append(log_probs)
            _, next_word = torch.max(log_probs, dim=1)
            preds.append(next_word)

            # select next input
            use_gt = np.random.rand() <= self.teacher_rate
            if (target_seq is not None) and self.training and use_gt:
                x = target_seq[:, i].unsqueeze(1)
            else:
                x = next_word.unsqueeze(1)

        return (torch.stack(logs_probs).permute(1, 0, 2),
                torch.stack(preds).permute(1, 0),
                torch.stack(attentions).squeeze(2).permute(1, 0, 2))

    def init_hidden(self, bs, device):
        return torch.zeros(2, self.n_layers, bs, self.hs,
                           dtype=torch.float, device=device)


class BahdanauAttention(nn.Module):
    def __init__(self, hs=256, key_size=None, query_size=None):
        super().__init__()

        # bidirectional lstm so key_size is 2*hidden_size
        key_size = 2 * hs if key_size is None else key_size
        query_size = hs if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hs, bias=False)
        self.decoder_linear = nn.Linear(query_size, hs, bias=False)
        self.energy_layer = nn.Linear(hs, 1, bias=False)

    def forward(self, dec_hs, enc_hss, value):
        # encoder states are already pre-computated
        dec_hs = self.decoder_linear(dec_hs)
        scores = self.energy_layer(torch.tanh(dec_hs + enc_hss))
        scores = scores.squeeze(2).unsqueeze(1)
        alphas = F.softmax(scores, dim=-1)
        context = torch.bmm(alphas, value)

        return context, alphas  # BS, 1, 2HS; BS, 1, M


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        position = torch.arange(max_len)
        a = torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        div_term = torch.exp(a).unsqueeze(1)

        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2, :] = torch.sin(position * div_term)
        pe[0, 1::2, :] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]
