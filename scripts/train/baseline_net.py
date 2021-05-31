#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Baseline net."""
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from string import digits, ascii_letters


class BaselineNet(nn.Module):
    def __init__(self,
                 height=64,
                 enc_hidden_size=256,
                 enc_num_layers=3,
                 enc_bidirectional=True,
                 emb_size=64,
                 max_len=150):
        super().__init__()
        self.max_len = max_len
        self.c2i, self.i2c = BaselineNet._build_alphabet()
        enc_out_channels = len(self.i2c)

        self.fe = FeatureExtractor()
        self.encoder = Encoder(input_size=self._get_encoder_input_size(height),
                               hidden_size=enc_hidden_size,
                               num_layers=enc_num_layers,
                               bidirectional=enc_bidirectional,
                               out_channels=enc_out_channels)
        self.decoder = AttentionDecoder(enc_out_channels,
                                        emb_size,
                                        sos_idx=self.c2i['<sos>'],
                                        eos_idx=self.c2i['<eos>'],
                                        max_len=self.max_len)
        self.ctc_loss = nn.CTCLoss(blank=self.c2i['<blank>'])

    def forward(self, x):
        x = self.fe(x)
        h = self.encoder(x)
        decoded = self.decoder(h)

        return decoded

    def calc_loss(self, log_probs, targets, targets_lens, preds):
        preds = preds.permute(1, 0)
        preds_lens = [self.max_len] * targets.size(0)
        last_sample = -1
        for sample, idx in torch.nonzero(preds == self.c2i['<eos>']):
            if sample != last_sample:
                last_sample = sample
                preds_lens[sample] = idx
        preds_lens = torch.tensor(preds_lens, dtype=torch.int64)

        return self.ctc_loss(log_probs, targets, preds_lens, targets_lens)

    @torch.no_grad()
    def _get_encoder_input_size(self, height):
        x = torch.rand(1, 1, height, 224)
        y = self.fe(x)

        return y.shape[1]

    @staticmethod
    def _build_alphabet():
        symbs = ['<blank>', '<sos>', '<eos>', ' ', '!', '"', '#', '&', '\'',
                 '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']
        i2c = symbs + list(digits) + list(ascii_letters)
        c2i = {c: idx for idx, c in enumerate(i2c)}

        return c2i, i2c


# dont forget about FPN
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(1, 8, (6, 4), (3, 2)),  # to fit 64-height images
            nn.LeakyReLU(),
            nn.Conv2d(8, 32, (6, 4), (1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d((4, 2), (4, 2)),
            nn.Conv2d(32, 64, (3, 3), (1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2), (1, 2))
        )

    def forward(self, x):
        y = self.fe(x)
        return y.squeeze(2)


class Encoder(nn.Module):
    def __init__(self,
                 input_size=64,
                 hidden_size=256,
                 num_layers=3,
                 bidirectional=True,
                 dropout=0.5,
                 out_channels=81):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            dropout=dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(1, out_channels, (256, 1), (1, 1)),
            # nn.LeakyReLU()
        )

    def forward(self, x):
        seq = x.permute(2, 0, 1)  # seq_len(width), bs, input_size(channels)

        hidden_states = []
        bs = seq.size(1)
        h, c = self.init_hidden(bs, seq.dtype, seq.device)
        for x_t in seq:
            _, (h, c) = self.lstm(x_t.unsqueeze(0), (h, c))
            hidden_states.append(h.view(self.num_layers,
                                        self.num_directions,
                                        bs,
                                        self.hidden_size)[-1, -1])

        y = torch.stack(hidden_states, dim=0)  # seq_len, bs, hidden_size
        y = y.permute(1, 2, 0)  # bs, seq_len(width), hidden_size
        y = y.unsqueeze(1)
        logits = self.conv(y).squeeze(2)  # bs, out_channels, width

        return torch.argmax(F.softmax(logits, dim=1), dim=1)

    def init_hidden(self, bs, dtype, device):
        num_directions = 2 if self.bidirectional else 1
        h_zeros = torch.zeros(self.num_layers * num_directions,
                              bs, self.hidden_size,
                              dtype=dtype, device=device)
        c_zeros = torch.zeros(self.num_layers * num_directions,
                              bs, self.hidden_size,
                              dtype=dtype, device=device)

        return h_zeros, c_zeros


class AttentionDecoder(nn.Module):
    def __init__(self,
                 enc_hidden_size=81,
                 emb_size=64,
                 hidden_size=256,
                 dropout=0.5,
                 sos_idx=1,
                 eos_idx=2,
                 max_len=150):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.emb_size = emb_size
        self.bidirectional = False
        self.num_layers = 1
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(enc_hidden_size, self.emb_size)
        self.pe = PositionalEncoder(emb_size)
        self.lstm = nn.LSTM(input_size=self.emb_size * 2,
                            hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.att = AttentionBahdanau(emb_size,
                                     self.hidden_size,
                                     emb_size)

        self.fc = nn.Linear(self.hidden_size, enc_hidden_size)

    def forward(self, enc_out):
        embedded = self.emb(enc_out)  # bs, width, emb_size
        embedded = self.pe(embedded)
        seq = embedded.permute(1, 0, 2)  # seq_len(width), bs, emb_size

        bs = seq.size(1)
        probs, preds = [], []
        h, c = self.init_hidden(bs, seq.dtype, seq.device)
        a = torch.zeros(bs, self.emb_size)
        y = self.emb(torch.ones(bs, dtype=torch.int64) * self.sos_idx)  # <sos>
        for i in range(self.max_len):
            y = torch.cat([y, a], dim=1)  # bs, emb_size * 2
            y, (h, c) = self.lstm(y.unsqueeze(0), (h, c))

            # attention
            att_weights = self.att(embedded, h.squeeze(0))  # bs, width, v_size
            a = torch.sum(embedded * att_weights, dim=1)  # bs, emb_size

            # calc probs
            y = self.fc(y.squeeze())  # bs, alphabet_size
            prob = F.softmax(y, dim=1)
            probs.append(torch.log(prob))  # for CTCLoss

            # check end of prediction
            pred = torch.argmax(prob, dim=1)
            preds.append(pred)
            if (pred == self.eos_idx).sum() == pred.shape[0]:
                break

            # lstm input preparation
            y = self.emb(pred)

        return torch.stack(probs), torch.stack(preds)

    def init_hidden(self, bs, dtype, device):
        num_directions = 2 if self.bidirectional else 1
        h_zeros = torch.zeros(self.num_layers * num_directions,
                              bs, self.hidden_size,
                              dtype=dtype, device=device)
        c_zeros = torch.zeros(self.num_layers * num_directions,
                              bs, self.hidden_size,
                              dtype=dtype, device=device)

        return h_zeros, c_zeros


class AttentionBahdanau(nn.Module):
    def __init__(self,
                 enc_out_size=64,
                 dec_out_size=256,
                 v_size=64):
        super().__init__()
        self.w_enc = nn.Linear(enc_out_size, v_size, bias=True)
        self.w_dec = nn.Linear(dec_out_size, v_size, bias=False)
        self.v = nn.Linear(v_size, v_size, bias=False)

    def forward(self, enc_emb, dec_emb):
        weighted_enc = self.w_enc(enc_emb).permute(1, 0, 2)
        weighted_dec = self.w_dec(dec_emb)
        summed = (weighted_enc + weighted_dec).permute(1, 0, 2)
        energy = self.v(torch.tanh(summed))

        return torch.softmax(energy, dim=1)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=150):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        a = torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        div_term = torch.exp(a) * position
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, x.size(1)]
