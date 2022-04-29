#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import torch

from torch import nn
from torch.nn import functional as F

from .conv_net import ConvNet6
from ctcdecode import CTCBeamDecoder


class BaselineNet(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 backbone_out=256,
                 n_layers=2,
                 dec_hs=256,
                 fe_dropout=0.15):
        super().__init__()

        self.c2i = c2i
        self.i2c = i2c
        alpb_size = len(self.i2c)

        self.fe = ConvNet6(out_channels=backbone_out, dropout=fe_dropout)
        self.rnn = nn.LSTM(input_size=backbone_out,
                           hidden_size=dec_hs,
                           num_layers=n_layers,
                           bidirectional=True)
        self.dropout = nn.Dropout(0.15)
        self.conv = nn.Conv2d(dec_hs * 2, alpb_size, (1, 1))

        self.loss_fn = nn.CTCLoss(zero_infinity=True)
        self.ctc_decoder = CTCBeamDecoder(self.i2c,
                                          beam_width=20,
                                          num_processes=4,
                                          log_probs_input=True)

    def forward(self, x):
        y = self.fe(x).squeeze(2).permute(2, 0, 1)
        y, _ = self.rnn(y)  # W, BS, 2HS
        y = y.permute(1, 2, 0).unsqueeze(2)  # BS, 2HS, 1, W
        logits = self.conv(self.dropout(y)).squeeze(2)  # BS, AS, W
        log_probs = F.log_softmax(logits, dim=1)  # BS, AS, W

        return logits, log_probs

    def calc_lens(self, logits):
        """Calculate lengths of predicted lines.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of size (bs, seq_len).

        Returns
        -------
        numpy.ndarray
            Array of (bs,) size with lengths.

        """
        preds = torch.argmax(logits, dim=1)
        probs_lens = [preds.size(1)] * preds.size(0)
        last_sample = -1

        for sample, idx in torch.nonzero(preds == self.c2i['é']):
            if sample != last_sample:
                last_sample = sample
                probs_lens[sample] = idx.item() + 1

        return probs_lens

    def calc_loss(self, logits, log_probs, targets, targets_lens):
        log_probs = log_probs.permute(2, 0, 1)

        probs_lens = torch.tensor(self.calc_lens(logits))
        probs_lens = torch.max(probs_lens.to(log_probs.device),
                               targets_lens)

        return self.loss_fn(log_probs, targets, probs_lens, targets_lens)

    def decode(self, log_probs):
        log_probs = log_probs.detach().permute(0, 2, 1)  # BS, W, AS
        decoded, _, _, lens = self.ctc_decoder.decode(log_probs.detach())

        decoded, lens = decoded[:, 0], lens[:, 0]  # get top score beam
        for idx, length in enumerate(lens):
            decoded[idx, length:] = self.c2i['_']

        return decoded, lens
