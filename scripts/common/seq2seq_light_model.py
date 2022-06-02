#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Seq2Seq model with 2D attention and no Encoder."""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .conv_net import get_backbone
from .layers import PositionalEncoder, PositionalEncoder2D, FlexibleLayerNorm


class Seq2seqLightModel(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 text_max_len,
                 backbone='conv_net6',
                 backbone_out=512,
                 dec_hs=256,
                 attn_sz=256,
                 emb_size=128,
                 dec_dropout=0.1,
                 pe=False,
                 teacher_rate=0.9,
                 fe_dropout=0.15,
                 expand_h=False,
                 gated=False):
        super().__init__()
        self.i2c = i2c
        self.c2i = c2i
        sos_idx = self.c2i['ś']
        alpb_size = len(self.i2c)

        self.backbone_out = backbone_out
        self.fe, self.backbone_out = get_backbone(
            backbone,
            out_channels=self.backbone_out,
            dropout=fe_dropout,
            expand_h=expand_h,
            gated=gated
        )
        pe_class = PositionalEncoder2D if expand_h else PositionalEncoder
        self.pe = pe_class(self.backbone_out) if pe else None
        self.decoder = Decoder(text_max_len=text_max_len,
                               sos_idx=sos_idx,
                               emb_sz=emb_size,
                               enc_hs=self.backbone_out,
                               dec_hs=dec_hs,
                               attn_sz=attn_sz,
                               alphabet_size=alpb_size,
                               dropout_p=dec_dropout,
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
        y = self.fe(x)  # (BS, 256, H, W)

        if self.pe is not None:
            y = self.pe(y)

        return self.decoder(y, target_seq)


class Decoder(nn.Module):
    def __init__(self,
                 text_max_len,
                 sos_idx=0,
                 emb_sz=128,
                 enc_hs=128,
                 dec_hs=256,
                 attn_sz=256,
                 alphabet_size=81,
                 dropout_p=0.1,
                 teacher_rate=0.9):
        super().__init__()
        print('========== Decoder args ==========')
        print('text_max_len: {}; sos_idx: {}; emb_sz: {}; enc_hs: {}; '
              'dec_hs: {}; alphabet_size: {}; dropout_p: {}; attn_sz: {}; '
              'teacher_rate: {};'.format(
            text_max_len, sos_idx, emb_sz, enc_hs, dec_hs, alphabet_size,
            dropout_p, attn_sz, teacher_rate
        ))

        self.sos_idx = sos_idx
        self.dropout_p = dropout_p
        self.max_len = text_max_len
        self.teacher_rate = teacher_rate
        self.dec_hs = dec_hs

        self.emb = nn.Embedding(alphabet_size, emb_sz)
        self.dropout = nn.Dropout(self.dropout_p)

        self.lstm = nn.LSTMCell(input_size=emb_sz + enc_hs,
                                hidden_size=self.dec_hs)

        self.attention = BahdanauAttention(enc_hs, self.dec_hs, attn_sz)
        self.linear = nn.Linear(self.dec_hs, alphabet_size)

    def forward_step(self, x, enc_out, weighted_enc_out, hc):
        dec_h = hc[0]  # BS, HS

        context, attn_probs = self.attention(dec_h, weighted_enc_out, enc_out)
        rnn_x = torch.cat([x, context], dim=1)
        h, c = self.lstm(rnn_x, hc)

        logits = self.linear(self.dropout(h))

        return logits, (h, c), attn_probs

    def forward(self, enc_out, target_seq=None):
        weighted_enc_out = self.attention.encoder_conv(enc_out)  # BS, A, H, W

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
                prev_embed, enc_out, weighted_enc_out, (h, c)
            )
            attentions.append(attn_probs)

            # BS, HS
            log_probs = F.log_softmax(output, dim=-1)
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
                torch.stack(attentions).permute(1, 0, 2, 3))

    def init_hidden(self, bs, device):
        return torch.zeros(2, bs, self.dec_hs,
                           dtype=torch.float, device=device)


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hs=256, dec_hs=384, attn_size=512):
        super().__init__()

        self.encoder_conv = nn.Conv2d(enc_hs, attn_size, (3, 3), (1, 1), (1, 1))
        self.decoder_linear = nn.Linear(dec_hs, attn_size)
        self.energy_layer = nn.Sequential(
            nn.Conv2d(attn_size, attn_size // 2, (3, 3), (1, 1), (1, 1)),
            FlexibleLayerNorm([-2, -1]),
            nn.ReLU(),
            nn.Conv2d(attn_size // 2, 1, (1, 1))
        )

    def forward(self, dec_h, weighted_enc_out, enc_out):
        weighted_dec_hs = self.decoder_linear(dec_h).unsqueeze(2).unsqueeze(3)
        pre_scores = torch.tanh(weighted_enc_out + weighted_dec_hs)

        scores = self.energy_layer(pre_scores)
        alphas = F.softmax(scores.flatten(1), dim=1).reshape(scores.shape)
        context = (enc_out * alphas).sum(dim=[2, 3])

        return context, alphas.squeeze(1)  # BS, enc_hs; BS, H, W
