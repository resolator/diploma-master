#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Network with custom attention."""
import torch
from torch import nn
from torch.nn import functional as F
from .conv_net import ConvNet6


class SegAttnModel(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 text_max_len=98,
                 backbone_out=256,
                 dec_dropout=0.15,
                 teacher_rate=1.0,
                 decoder_type='attn_rnn',
                 fe_dropout=0.0,
                 emb_size=256):
        super().__init__()

        self.c2i = c2i
        self.i2c = i2c
        self.sos_idx = self.c2i['ś']

        self.backbone_out = backbone_out
        self.fe = ConvNet6(out_channels=self.backbone_out,
                           dropout=fe_dropout)
        decoder_args = {'c2i': c2i,
                        'i2c': i2c,
                        'x_size': self.backbone_out,
                        'sos_idx': self.sos_idx,
                        'dropout': dec_dropout,
                        'text_max_len': text_max_len,
                        'teacher_rate': teacher_rate,
                        'emb_size': emb_size}

        if decoder_type == 'attn_rnn':
            self.decoder = AttnRNNDecoder(**decoder_args)
        else:
            self.decoder = AttnFCDecoder(**decoder_args)

        self.loss_fn = nn.NLLLoss(reduction='none', ignore_index=self.sos_idx)

    def forward(self, x, target_seq=None):
        fm = self.fe(x)
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
        attn_size = 256
        self.attention = Attention(h_dec_size=self.hs,
                                   channels=x_size,
                                   out_size=attn_size)
        self.rnn = nn.LSTMCell(input_size=emb_size + attn_size,
                               hidden_size=self.hs)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.hs, self.hs // 2),
            nn.BatchNorm1d(self.hs // 2),
            nn.ReLU(),
            nn.Linear(self.hs // 2, alpb_size)
        )

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
            y_emb = self.dropout(self.emb(y))  # (BS, 256)
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


class AttnFCDecoder(nn.Module):
    def __init__(self,
                 c2i,
                 i2c,
                 x_size=256,
                 h_size=512,
                 emb_size=256,
                 dropout=0.15,
                 sos_idx=1,
                 text_max_len=98,
                 teacher_rate=1.0):
        super().__init__()
        print('========== AttnFCDecoder args ==========')
        print('x_size: {}; h_size: {}; emb_size: {}; dropout: {}; '
              'text_max_len: {};'.format(
            x_size, h_size, emb_size, dropout, text_max_len
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
        self.attention = Attention(h_dec_size=h_size,
                                   channels=x_size,
                                   out_size=emb_size)
        self.rnn = nn.LSTMCell(input_size=emb_size,
                               hidden_size=self.hs)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(h_size, h_size)
        self.attn_linear = nn.Linear(emb_size, h_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(h_size + h_size)
        self.last_fc = nn.Linear(h_size + h_size, alpb_size)

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
            h, c = self.rnn(y_emb, (h, c))

            attn, heat_map = self.attention(h, fm)
            attentions.append(heat_map)

            weighted_attn = self.attn_linear(self.attn_dropout(attn))
            pre_logits = self.fc(self.dropout(h))
            pre_logits = torch.cat([pre_logits, weighted_attn], dim=1)
            pre_logits = self.relu(self.bn(pre_logits))
            step_logits = self.last_fc(pre_logits)

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

        self.h_dec_proc = nn.Linear(h_dec_size, channels)
        self.summed_x_proc = nn.Sequential(
            nn.Conv2d(2 * channels, channels, (1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels // 2, (1, 1)),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, (1, 1))
        )
        self.attended_proc = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, (1, 1))
        )
        self.summed_proc = nn.Sequential(
            nn.BatchNorm1d(channels),
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
        repeated_h_dec = weighted_h_dec.repeat(1, 1, fm.size(2), fm.size(3))

        concated = torch.cat([fm, repeated_h_dec], dim=1)  # BS, 2C, H, W
        attn_map = self.summed_x_proc(concated).squeeze(1)  # BS, H, W

        heat_map = F.softmax(attn_map.flatten(1),
                             dim=1).reshape(attn_map.shape)

        attn = fm * heat_map.unsqueeze(1)
        attn_proc = self.attended_proc(attn)
        summed = attn_proc.sum(dim=[2, 3])
        context = self.summed_proc(summed)

        return context, heat_map
