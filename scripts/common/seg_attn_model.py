#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Network with custom attention."""
import torch
from torch import nn
from torch.nn import functional as F


class SegAttnModel(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 text_max_len=98,
                 backbone_out=256,
                 dropout=0.15):
        super().__init__()

        self.c2i = c2i
        self.i2c = i2c
        self.sos_idx = self.c2i['ś']

        self.backbone_out = backbone_out
        self.backbone = FeatureExtractor(out_channels=self.backbone_out)
        self.decoder = Decoder(c2i=c2i,
                               i2c=i2c,
                               x_size=self.backbone_out,
                               sos_idx=self.sos_idx,
                               dropout=dropout,
                               text_max_len=text_max_len)

        self.loss_fn = nn.NLLLoss(reduction='none', ignore_index=self.sos_idx)

    def forward(self, x, target_seq=None):
        fm = self.backbone(x)
        log_probs, preds, attentions = self.decoder(fm, target_seq)

        return log_probs, preds, attentions

    def calc_loss(self, logs_probs, targets, targets_lens):
        loss = self.loss_fn(logs_probs, targets)

        # lets mask the loss
        mask = torch.zeros(loss.shape, dtype=torch.int64)

        for i, t_len in enumerate(targets_lens):
            mask[i, :t_len] = 1

        loss = torch.mean(loss[mask])

        return loss


class FeatureExtractor(nn.Module):
    def __init__(self, out_channels=256):
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

            nn.Conv2d(256, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Dropout(0.15)
        )

    def forward(self, x):
        # x.shape == BS, 1, 64, W
        # return self.fe(x).squeeze(2)  # BS, 256, W // 4
        return self.fe(x)  # BS, 256, 1, W // 4


class Decoder(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 x_size=256,
                 h_size=512,
                 emb_size=256,
                 dropout=0.15,
                 sos_idx=1,
                 text_max_len=98):
        super().__init__()
        print('========== Decoder args ==========')
        print('x_size: {}; h_size: {}; emb_size: {}; dropout: {}; '
              'text_max_len: {};'.format(
            x_size, h_size, emb_size, dropout, text_max_len
        ))

        self.text_max_len = text_max_len
        self.c2i = c2i
        self.i2c = i2c
        alpb_size = len(self.i2c)
        self.sos_idx = sos_idx

        self.n_layers = 1
        self.hs = h_size

        self.emb = nn.Embedding(alpb_size, emb_size)
        self.attention = Attention(h_dec_size=h_size,
                                   channels=x_size,
                                   out_size=emb_size,
                                   dropout=dropout)
        self.rnn = nn.LSTMCell(input_size=emb_size + x_size,
                               hidden_size=self.hs)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(h_size, alpb_size)

    def forward_step(self, y_emb, h, fm):
        h_attn = h[0]  # (BS, emb_size)
        z, heat_map = self.attention(h_attn, fm)  # (BS, emb_size)
        x = torch.cat([y_emb, z], dim=1)  # BS, 2 * emb_size

        return self.rnn(x, h), heat_map  # 2, BS, HS

    def forward(self, fm, target_seq=None):
        bs = fm.size(0)
        h, c = self.init_hidden(bs, fm.device)
        y = torch.ones(
            (bs,),
            dtype=torch.int64,
            device=fm.device
        ) * self.sos_idx

        attentions, log_probs, preds = [], [], []
        for i in range(self.text_max_len):
            y_emb = self.emb(y)  # (BS, 256)
            (h, c), heat_map = self.forward_step(y_emb, (h, c), fm)
            attentions.append(heat_map)

            step_logits = self.fc(self.dropout(h))

            step_log_probs = F.log_softmax(step_logits, dim=-1)
            log_probs.append(step_log_probs)

            _, pred = torch.max(step_log_probs, dim=1)
            preds.append(pred)

            # select next input
            if (target_seq is not None) and self.training:
                y = target_seq[:, i]
            else:
                y = pred

        # BS, C, text_max_len
        # BS, text_max_len
        # BS, text_max_len, H, W
        return (torch.stack(log_probs).permute(1, 2, 0),
                torch.stack(preds).permute(1, 0),
                torch.stack(attentions).permute(1, 0, 2, 3))

    def init_hidden(self, bs, device):
        return torch.zeros(2, bs, self.hs,
                           dtype=torch.float, device=device)


class Attention(nn.Module):
    def __init__(self, h_dec_size, channels, out_size=256, dropout=0.15):
        super().__init__()
        print('========== Attention args ==========')
        print('h_dec_size: {}; channels: {}; out_size: {}; dropout: {};'.format(
            h_dec_size, channels, out_size, dropout
        ))

        self.fm_conv = nn.Conv2d(channels, out_size, (1, 1))
        self.fm_dropout = nn.Dropout(dropout)

        self.h_dec_linear = nn.Linear(h_dec_size, out_size)
        self.h_dec_dropout = nn.Dropout(dropout)

    def forward(self, h_dec, fm):
        """Forward pass.

        Hidden size in the h_dec must be equal to the number of channels
        in the fm.

        Parameters
        ----------
            h_dec : torch.tensor
                Tensor of shape (BS, HS).
            fm : torch.tensor
                Features map of shape (BS, C, H, W).

        Returns
        -------
        (torch.tensor, torch.tensor)
            A context vector of shape (BS, attn_size) and heat map.

        """
        assert len(fm.shape) == 4, "invalid number of shape for fm"
        assert len(h_dec.shape) == 2, "invalid number of shape for h_dec"

        attn_map = torch.zeros(fm.size(0), fm.size(2), fm.size(3),
                               device=fm.device)
        weighted_h_dec = self.h_dec_dropout(
            self.h_dec_linear(h_dec)
        ).unsqueeze(-1).unsqueeze(-1)
        weighted_fm = self.fm_dropout(self.fm_conv(fm))

        attn_map = (weighted_fm * weighted_h_dec).sum(dim=1)
        heat_map = F.softmax(attn_map.flatten(1),
                               dim=1).reshape(attn_map.shape)

        attn = fm * heat_map.unsqueeze(1)
        context = attn.sum(dim=[2, 3])

        return context, heat_map
