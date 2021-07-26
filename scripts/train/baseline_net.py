#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Baseline net."""
import timm
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from string import digits, ascii_letters


class BaselineNet(nn.Module):
    def __init__(self,
                 height=64,
                 enc_hs=256,
                 dec_hs=256,
                 enc_n_layers=3,
                 enc_bidirectional=True,
                 max_len=300,
                 teacher_ratio=0.5):
        super().__init__()
        self.max_len = max_len
        self.teacher_ratio = teacher_ratio
        self.c2i, self.i2c = BaselineNet._build_alphabet()
        enc_output_size = len(self.i2c)

        self.fe = FeatureExtractor()
        self.encoder = Encoder(input_size=self._get_encoder_input_size(height),
                               hidden_size=enc_hs,
                               num_layers=enc_n_layers,
                               bidirectional=enc_bidirectional,
                               output_size=enc_output_size)
        self.decoder = AttentionDecoder(enc_os=enc_output_size,
                                        hidden_size=dec_hs,
                                        sos_idx=self.c2i['<sos>'],
                                        eos_idx=self.c2i['<eos>'],
                                        max_len=self.max_len)

    def forward(self, x, target_seq=None):
        x = self.fe(x)
        enc_out = self.encoder(x)
        teacher = np.random.random() < self.teacher_ratio
        logits, loss = self.decoder(enc_out,
                                    target_seq=target_seq,
                                    teacher=teacher)

        return logits, loss

    @torch.no_grad()
    def _get_encoder_input_size(self, height):
        x = torch.rand(1, 1, height, 224)
        y = self.fe(x)

        return y.shape[1]

    @staticmethod
    def _build_alphabet():
        symbs = ['<b>', '<sos>', '<eos>', ' ', '!', '"', '#', '&', '\'',
                 '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']
        i2c = symbs + list(digits) + list(ascii_letters)
        c2i = {c: idx for idx, c in enumerate(i2c)}

        return c2i, i2c


# dont forget about FPN
class FeatureExtractor(nn.Module):
    def __init__(self, resnet=False):
        super().__init__()
        if resnet:
            self.fe = timm.create_model('resnet18',
                                        num_classes=0,
                                        global_pool='',
                                        in_chans=1,
                                        pretrained=True)
        else:
            self.fe = nn.Sequential(
                nn.Conv2d(1, 8, (6, 4), (3, 2)),  # to fit 64-height images
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.Conv2d(8, 32, (6, 4), (1, 1)),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.MaxPool2d((4, 2), (4, 2)),
                nn.Conv2d(32, 64, (3, 3), (1, 1)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d((1, 2), (1, 2))
            )

    def forward(self, x):
        y = self.fe(x)
        return torch.mean(y, dim=2).squeeze(2)  # BS, C, W


class Encoder(nn.Module):
    def __init__(self,
                 input_size=64,
                 hidden_size=256,
                 num_layers=3,
                 bidirectional=True,
                 dropout=0.1,
                 output_size=81):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(1, output_size, (hidden_size * 2, 1), (1, 1)),
            # nn.LeakyReLU()
        )

    def forward(self, x):
        # BS, C, W
        seq = x.permute(2, 0, 1)  # seq_len(W), BS, input_size(C)
        hs, _ = self.lstm(seq)  # seq_len(W), BS, HS * 2
        hs = hs.permute(1, 2, 0)  # BS, HS * 2, seq_len(W)
        hs = hs.unsqueeze(1)  # BS, 1, HS * 2, W
        logits = self.conv(hs).squeeze(2)  # BS, out_C, W

        return torch.argmax(F.softmax(logits, dim=1), dim=1)  # BS, W


class AttentionDecoder(nn.Module):
    def __init__(self,
                 enc_os=81,
                 hidden_size=256,
                 dropout=0.1,
                 sos_idx=1,
                 eos_idx=2,
                 max_len=300):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.hidden_size = hidden_size
        self.criterion = nn.CrossEntropyLoss()

        self.emb = nn.Embedding(enc_os, self.hidden_size)
        self.pe = PositionalEncoder(self.hidden_size)
        self.lstm_cell = nn.LSTMCell(input_size=self.hidden_size,
                                     hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # attention
        self.att = nn.Linear(hidden_size * 2, max_len)
        self.att_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.fc = nn.Linear(self.hidden_size, enc_os)

    def forward(self, enc_out, target_seq=None, teacher=False):
        encoded = self.emb(enc_out)  # BS, W, HS
        # encoded = self.pe(encoded)  # BS, W, HS

        pad_tensor = torch.zeros(encoded.size(0),
                                 self.max_len - encoded.size(1),
                                 encoded.size(2))
        padded_enc_out = torch.cat([encoded, pad_tensor], dim=1)

        seq = encoded.permute(1, 0, 2)  # seq_len(W), BS, HS
        bs = seq.size(1)

        h_dec, c = self.init_hidden(bs, seq.device)
        x_dec = torch.ones(
            bs,
            dtype=torch.int64,
            device=seq.device
        ) * self.sos_idx  # BS (<sos>)

        seq_logits = []
        loss = 0.0
        for i in range(self.max_len):
            x_emb = self.emb(x_dec)  # BS, HS
            x_emb = self.dropout(x_emb)

            # attention
            att_x = torch.cat([x_emb, h_dec], 1)  # BS, 2HS
            energy = self.att(att_x)  # BS, max_len
            att_w = F.softmax(energy, dim=1)

            # (1, max_len) X (max_len, HS)
            a = torch.bmm(att_w.unsqueeze(1), padded_enc_out)  # BS, 1, HS
            a = a.squeeze(1)  # BS, HS
            x = torch.cat([x_emb, a], dim=1)  # BS, 2HS
            x = F.relu(self.att_combine(x))  # BS, HS

            h_dec, c = self.lstm_cell(x, (h_dec, c))  # BS, HS
            logits = self.fc(h_dec)  # BS, OS
            seq_logits.append(logits)

            # use targets as the next input?
            if target_seq is not None:
                loss += self.criterion(logits, target_seq[i])

                if teacher:
                    x_dec = target_seq[i]  # BS
                else:
                    x_dec = torch.argmax(logits, dim=1)  # BS

            # ended sequence?
            if (x_dec == self.eos_idx).sum() == x_dec.shape[0]:
                break

        return torch.stack(seq_logits), loss  # seq_len(W), BS, OS

    def init_hidden(self, bs, device):
        h_zeros = torch.zeros(bs, self.hidden_size,
                              dtype=torch.float, device=device)
        c_zeros = torch.zeros(bs, self.hidden_size,
                              dtype=torch.float, device=device)

        return h_zeros, c_zeros


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=1000):
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
