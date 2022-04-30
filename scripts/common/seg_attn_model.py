#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Network with custom attention."""
import timm
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .conv_net import ConvNet6


class SegAttnModel(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 text_max_len=98,
                 backbone='custom',
                 backbone_out=256,
                 dec_hs=512,
                 dec_dropout=0.15,
                 teacher_rate=1.0,
                 fe_dropout=0.0,
                 emb_size=256,
                 pos_enc=False):
        super().__init__()

        self.c2i = c2i
        self.i2c = i2c
        self.sos_idx = self.c2i['ś']

        assert backbone in ['custom', 'resnet18', 'resnet34', 'efficientnet_b0']
        if backbone == 'custom':
            self.backbone_out = backbone_out
            self.fe = ConvNet6(out_channels=self.backbone_out,
                               dropout=fe_dropout)
        else:
            self.fe = timm.create_model(backbone,
                                        pretrained=True,
                                        in_chans=1,
                                        num_classes=0,
                                        global_pool='')
            self.backbone_out = self.fe.num_features

        self.pe = PositionalEncoder(self.backbone_out) if pos_enc else None
        decoder_args = {'c2i': c2i,
                        'i2c': i2c,
                        'x_size': self.backbone_out,
                        'h_size': dec_hs,
                        'sos_idx': self.sos_idx,
                        'dropout': dec_dropout,
                        'text_max_len': text_max_len,
                        'teacher_rate': teacher_rate,
                        'emb_size': emb_size}
        self.decoder = AttnRNNDecoder(**decoder_args)
        self.loss_fn = nn.NLLLoss(reduction='none', ignore_index=self.sos_idx)

    def forward(self, x, target_seq=None):
        fm = self.fe(x)

        if self.pe is not None:
            fm = self.pe(fm)

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


class AttnRNNDecoder(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 x_size=256,
                 h_size=512,
                 emb_size=256,
                 attn_size=256,
                 dropout=0.15,
                 sos_idx=1,
                 text_max_len=98,
                 teacher_rate=1.0):
        super().__init__()
        print('========== AttnRNNDecoder args ==========')
        print('x_size: {}; h_size: {}; emb_size: {}; dropout: {}; tr: {} '
              'text_max_len: {};'.format(
            x_size, h_size, emb_size, dropout, teacher_rate, text_max_len
        ))

        self.teacher_rate = teacher_rate
        self.text_max_len = text_max_len
        self.c2i = c2i
        self.i2c = i2c
        alpb_size = len(self.i2c)
        self.sos_idx = sos_idx

        self.n_layers = 1
        self.hs = h_size

        self.emb = nn.Embedding(alpb_size, emb_size)
        self.emb_postproc = nn.Sequential(
            FlexibleLayerNorm(-1),
            nn.Dropout(dropout)
        )
        self.attention = Attention(h_dec_size=self.hs,
                                   channels=x_size,
                                   out_size=attn_size)
        self.cat_linear = nn.Sequential(
            nn.Linear(emb_size + attn_size, emb_size + attn_size),
            FlexibleLayerNorm(-1)
        )
        self.rnn = nn.LSTMCell(input_size=emb_size + attn_size,
                               hidden_size=self.hs)
        self.fc = nn.Sequential(
            nn.Linear(self.hs, self.hs // 2),
            FlexibleLayerNorm(-1),
            nn.ReLU(),
            nn.Linear(self.hs // 2, alpb_size)
        )

    def forward_step(self, y_emb, h, fm):
        h_attn = h[0]  # (BS, emb_size)
        context, heat_map = self.attention(h_attn, fm)  # (BS, emb_size)
        x = torch.cat([y_emb, context], dim=1)  # BS, emb_size + attn_size
        x = self.cat_linear(x)

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
            y_emb = self.emb_postproc(self.emb(y))  # (BS, 256)
            (h, c), heat_map = self.forward_step(y_emb, (h, c), fm)
            attentions.append(heat_map)

            step_logits = self.fc(h)

            step_log_probs = F.log_softmax(step_logits, dim=-1)
            log_probs.append(step_log_probs)

            _, pred = torch.max(step_log_probs, dim=1)
            preds.append(pred)

            # select next input
            teach = torch.rand(1)[0].item() < self.teacher_rate
            if (target_seq is not None) and self.training and teach:
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
    def __init__(self, h_dec_size, channels, out_size=256):
        super().__init__()
        print('========== Attention args ==========')
        print('h_dec_size: {}; channels: {}; out_size: {};'.format(
            h_dec_size, channels, out_size
        ))

        self.h_dec_proc = nn.Sequential(
            FlexibleLayerNorm(-1),
            nn.Linear(h_dec_size, channels),
            nn.ReLU()
        )
        self.summed_x_proc = nn.Sequential(
            nn.Conv2d(channels, channels // 2, (1, 1)),
            FlexibleLayerNorm([-3, -2, -1]),  # (C, H, W)
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, (1, 1)),
        )
        self.attended_proc = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, (1, 1))
        )
        self.summed_proc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, out_size)
        )

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

        weighted_h_dec = self.h_dec_proc(h_dec).unsqueeze(-1).unsqueeze(-1)

        summed = fm + weighted_h_dec  # BS, C, H, W
        attn_map = self.summed_x_proc(summed).squeeze(1)  # BS, H, W

        heat_map = F.softmax(attn_map.flatten(1),
                             dim=1).reshape(attn_map.shape)

        attn = fm * heat_map.unsqueeze(1)  # BS, C, H, W
        attn_proc = self.attended_proc(attn)
        summed = attn_proc.sum(dim=[2, 3])
        context = self.summed_proc(summed)

        return context, heat_map


class FlexibleLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """Flexible LayerNorm.

        Parameters
        ----------
        dim : int, list
            Number of dimensions to perform the norm, by default -1
        eps : float
            Epsilon, by default 1e-5

        """
        super().__init__()

        self.eps = eps
        self.dim = dim

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return (x - mean) / (std + self.eps)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=512):
        """Positional encoding layer.

        Parameters
        ----------
        d_model : int
            Number of channels for the output from layer before this.
        max_len : int, optional
            Maximal length of the layer before, by default 512.

        """
        super().__init__()

        position = torch.arange(max_len)
        a = torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        div_term = torch.exp(a).unsqueeze(1)

        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2, :] = torch.cos(position * div_term)
        pe[0, 1::2, :] = torch.sin(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if len(x.shape) == 3:
            return x + self.pe[:, :, :x.size(-1)]  # BS, C, W
        elif len(x.shape) == 4:
            return x + self.pe[:, :, :x.size(-1)].unsqueeze(2)  # BS, C, H, W
        else:
            raise AssertionError(
                'input shape must be 3 (for sequences) or 4 (for images).'
            )
