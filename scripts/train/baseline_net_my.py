#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Baseline net."""
import timm
import torch

import numpy as np
import torch.nn.functional as F
import tensorflow as tf

from torch import nn
from utils import build_alphabet


class BaselineNet(nn.Module):
    def __init__(self,
                 height=64,
                 enc_hs=256,
                 dec_hs=256,
                 enc_n_layers=3,
                 enc_bidirectional=True,
                 max_len=150,
                 teacher_ratio=0.5):
        super().__init__()
        self.max_len = max_len
        self.c2i, self.i2c = build_alphabet()
        enc_out_channels = len(self.i2c)

        self.fe = FeatureExtractor()
        self.encoder = Encoder(input_size=self._get_encoder_input_size(height),
                               hidden_size=enc_hs,
                               num_layers=enc_n_layers,
                               bidirectional=enc_bidirectional,
                               out_channels=enc_out_channels)
        self.decoder = AttentionDecoder(enc_out_channels,
                                        64,
                                        hidden_size=dec_hs,
                                        sos_idx=self.c2i['ś'],
                                        eos_idx=self.c2i['é'],
                                        max_len=self.max_len,
                                        teacher_ratio=teacher_ratio)
        self.ctc_loss = nn.CTCLoss(self.c2i['ƀ'], 'mean', True)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target_seq=None, target_lens=None):
        x = self.fe(x)
        h = self.encoder(x)
        decoded = self.decoder(h,
                               target_seq=target_seq,
                               target_lens=target_lens)

        return decoded

    def calc_loss(self, logits, targets, targets_lens, preds, ce=False):
        if ce:
            logits = logits.permute(1, 2, 0)
            loss = self.ce_loss(logits, targets)

            # lets mask the loss
            mask = torch.zeros(loss.shape, dtype=torch.int64)

            for i, t_len in enumerate(targets_lens):
                mask[i, :t_len] = 1

            loss = torch.mean(loss[mask])

        else:
            probs = torch.log(F.softmax(logits, dim=-1))
            preds_lens = torch.from_numpy(self.calc_preds_lens(preds)).to(
                probs.device)
            loss = self.ctc_loss(probs, targets, preds_lens, targets_lens)

        return loss

    @torch.no_grad()
    def _get_encoder_input_size(self, height):
        x = torch.rand(1, 1, height, 224)
        y = self.fe(x)

        return y.shape[1]

    def calc_preds_lens(self, preds):
        """Calculate lengths of predicted lines.

        Parameters
        ----------
        preds : torch.Tensor
            Tensor of size (seq_len, bs).

        Returns
        -------
        numpy.ndarray
            Array of (bs,) size with lengths.

        """
        preds = preds.permute(1, 0)  # bs, seq_len
        bs = preds.size(0)
        preds_lens = [self.max_len] * bs
        last_sample = -1
        for sample, idx in torch.nonzero(preds == self.c2i['é']):
            if sample != last_sample:
                last_sample = sample
                preds_lens[sample] = idx

        return np.array(preds_lens, dtype=np.int64)

    @torch.no_grad()
    def decode_ctc_beam_search(self, logits, lens, beam_width=16):
        logits = logits.detach().cpu()

        res = []
        for i in range(logits.size(1)):  # iterate over batch size
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                logits[:, i].unsqueeze(1).numpy(),
                [lens[i]],
                beam_width,
                1
            )
            res.append(decoded[0].values.numpy())

        return res


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
        return torch.mean(y, dim=2).squeeze(2)


class Encoder(nn.Module):
    def __init__(self,
                 input_size=64,
                 hidden_size=256,
                 num_layers=3,
                 bidirectional=True,
                 dropout=0.1,
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
        h, c = self.init_hidden(bs, seq.device)
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

    def init_hidden(self, bs, device):
        return torch.zeros(2, self.num_layers * self.num_directions,
                           bs, self.hidden_size,
                           dtype=torch.float, device=device)


class AttentionDecoder(nn.Module):
    def __init__(self,
                 enc_hidden_size=81,
                 emb_size=64,
                 hidden_size=256,
                 dropout=0.1,
                 sos_idx=1,
                 eos_idx=2,
                 max_len=150,
                 teacher_ratio=0.5):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.emb_size = emb_size
        self.num_directions = 1
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.teacher_ratio = teacher_ratio

        self.emb = nn.Embedding(enc_hidden_size, self.emb_size)
        self.pe = PositionalEncoder(emb_size)
        self.lstm = nn.LSTM(input_size=self.emb_size * 2,
                            hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.att = AttentionBahdanau(emb_size,
                                     self.hidden_size,
                                     emb_size)

        self.fc = nn.Linear(self.hidden_size, enc_hidden_size)

    def forward(self, enc_out, target_seq=None, target_lens=None):
        embedded = self.emb(enc_out)  # bs, width, emb_size
        embedded = self.pe(embedded)
        seq = embedded.permute(1, 0, 2)  # seq_len(width), bs, emb_size
        cur_max_len = torch.max(target_lens)

        bs = seq.size(1)
        logits, preds = [], []
        h, c = self.init_hidden(bs, seq.device)
        att_weights = self.att(embedded, h.squeeze(0))  # bs, width, v_size
        a = torch.sum(embedded * att_weights, dim=1)  # bs, emb_size
        y = self.emb(torch.ones(bs,
                                dtype=torch.int64,
                                device=seq.device) * self.sos_idx)  # <sos>
        for i in range(self.max_len):
            y = torch.cat([y, a], dim=1)  # bs, emb_size * 2
            y, (h, c) = self.lstm(y.unsqueeze(0), (h, c))

            # attention
            att_weights = self.att(embedded, h.squeeze(0))  # bs, width, v_size
            a = torch.sum(embedded * att_weights, dim=1)  # bs, emb_size

            # calc probs
            y = self.fc(y.squeeze(0))  # bs, alphabet_size
            logits.append(y)

            # check end of prediction
            pred = torch.argmax(y, dim=-1)
            preds.append(pred)
            if (pred == self.eos_idx).sum() == pred.shape[0]:
                if len(logits) >= cur_max_len:
                    break

            # lstm input preparation
            if target_seq is not None:
                teacher = np.random.random() < self.teacher_ratio

                if teacher and self.training:
                    pred = target_seq[:, i]

            y = self.emb(pred)

        diff = self.max_len - len(logits)
        if diff > 0:
            eos_logit = torch.zeros_like(logits[0])
            eos_logit[self.eos_idx] = 100
            append_logits = [torch.clone(eos_logit) for _ in range(diff)]

            eos_pred = torch.argmax(eos_logit, dim=-1)
            append_preds = [torch.clone(eos_pred) for _ in range(diff)]

            logits.extend(append_logits)
            preds.extend(append_preds)

        return torch.stack(logits), torch.stack(preds)

    def init_hidden(self, bs, device):
        return torch.zeros(2, self.num_layers * self.num_directions,
                           bs, self.hidden_size,
                           dtype=torch.float, device=device)


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

        return F.softmax(energy, dim=1)


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
