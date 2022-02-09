#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Seq2Seq model with Bahdanau attention."""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .conv_net import ConvNet6


class Seq2seqModel(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 text_max_len,
                 enc_hs=128,
                 emb_size=128,
                 enc_n_layers=1,
                 dropout_p=0.1,
                 pe=False,
                 teacher_rate=0.9,
                 fe_dropout=0.15):
        super().__init__()
        self.i2c = i2c
        self.c2i = c2i
        sos_idx = self.c2i['ś']
        alpb_size = len(self.i2c)

        self.fe = ConvNet6(dropout=fe_dropout)
        self.encoder = Encoder(input_sz=256,
                               hs=enc_hs,
                               n_layers=enc_n_layers)
        self.pe = PositionalEncoder(enc_hs * 2) if pe else None
        self.decoder = Decoder(text_max_len=text_max_len,
                               sos_idx=sos_idx,
                               enc_hs=enc_hs,
                               emb_size=emb_size,
                               alphabet_size=alpb_size,
                               dropout_p=dropout_p,
                               teacher_rate=teacher_rate)
        self.loss_f = nn.NLLLoss(reduction='none', ignore_index=sos_idx)

    def calc_loss(self, logs_probs, targets, targets_lens):
        logs_probs = logs_probs.permute(0, 2, 1)  # BS, AS, W
        loss = self.loss_f(logs_probs, targets)

        # lets mask the loss
        mask = torch.zeros(loss.shape, dtype=torch.int64)

        for i, t_len in enumerate(targets_lens):
            mask[i, :t_len] = 1

        return torch.mean(loss[mask])

    def forward(self, x, target_seq=None):
        y = self.fe(x).squeeze(2)
        y = self.encoder(y)

        if self.pe is not None:
            y = self.pe(y)

        return self.decoder(y, target_seq)


class Encoder(nn.Module):
    def __init__(self, input_sz=256, hs=256, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_sz,
                            hs,
                            num_layers=n_layers,
                            bidirectional=True)

    def forward(self, x):
        # BS, C, W
        x = x.permute(2, 0, 1)  # W, BS, C
        y, _ = self.lstm(x)  # W, BS, 2HS

        return y.permute(1, 2, 0)  # BS, 2HS, W


class Decoder(nn.Module):
    def __init__(self,
                 text_max_len,
                 sos_idx=0,
                 emb_size=256,
                 enc_hs=256,
                 alphabet_size=81,
                 dropout_p=0.1,
                 teacher_rate=0.9):
        super().__init__()
        self.sos_idx = sos_idx
        self.hs = emb_size + enc_hs
        self.alphabet_size = alphabet_size
        self.dropout_p = dropout_p
        self.emb_size = emb_size
        self.max_len = text_max_len
        self.teacher_rate = teacher_rate

        self.emb = nn.Embedding(alphabet_size, self.emb_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTMCell(input_size=self.hs + enc_hs,
                                hidden_size=self.hs)
        self.attention = BahdanauAttention(self.hs)
        self.linear_1 = nn.Linear(self.emb_size + self.hs + self.hs, self.hs)
        self.linear_2 = nn.Linear(self.hs, self.alphabet_size)

    def forward_step(self, x, enc_out, proj_key, hc):
        query = hc[0]  # BS, HS

        context, attn_probs = self.attention(query, proj_key, enc_out)
        rnn_x = torch.cat([x, context], dim=1)
        h, c = self.lstm(rnn_x, hc)

        pre_output = torch.cat([x, h, context], dim=1)
        pre_output = self.dropout(pre_output)
        logits = self.linear_1(pre_output)

        return logits, (h, c), attn_probs

    def forward(self, enc_out, target_seq=None):
        proj_key = self.attention.key_layer(enc_out)

        bs = enc_out.shape[0]
        h, c = self.init_hidden(bs, enc_out.device)
        x = torch.ones(
            bs,
            dtype=torch.int64,
            device=enc_out.device
        ) * self.sos_idx

        logs_probs, preds, attentions = [], [], []
        for i in range(self.max_len):
            prev_embed = self.emb(x)

            output, (h, c), attn_probs = self.forward_step(
                prev_embed, enc_out, proj_key, (h, c)
            )
            attentions.append(attn_probs)

            # BS, HS
            log_probs = F.log_softmax(self.linear_2(output), dim=-1)
            logs_probs.append(log_probs)
            _, next_word = torch.max(log_probs, dim=1)
            preds.append(next_word)

            # select next input
            use_gt = np.random.rand() <= self.teacher_rate
            if (target_seq is not None) and self.training and use_gt:
                x = target_seq[:, i]
            else:
                x = next_word

        return (torch.stack(logs_probs).permute(1, 0, 2),
                torch.stack(preds).permute(1, 0),
                torch.stack(attentions).squeeze(2).permute(1, 0, 2))

    def init_hidden(self, bs, device):
        return torch.zeros(2, bs, self.hs,
                           dtype=torch.float, device=device)


class BahdanauAttention(nn.Module):
    def __init__(self, hs=256):
        super().__init__()

        self.key_layer = nn.Conv1d(hs, hs, 1)
        self.decoder_linear = nn.Linear(hs, hs)
        self.energy_layer = nn.Conv1d(hs, 1, 1)

    def forward(self, dec_hs, enc_hss, value):
        # encoder states are already pre-computated
        dec_hs = self.decoder_linear(dec_hs).unsqueeze(2)
        pre_scores = torch.tanh(enc_hss + dec_hs)
        scores = self.energy_layer(pre_scores)
        alphas = F.softmax(scores, dim=-1)
        context = torch.bmm(value, alphas.permute(0, 2, 1))

        return context.squeeze(2), alphas.squeeze(1)  # BS, 2HS; BS, W


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
