#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Baseline net."""
import timm
import torch

from torch import nn


class BaselineNet(nn.Module):
    def __init__(self,
                 height,
                 enc_hidden_size=512,
                 enc_num_layers=1):
        super().__init__()

        self.fe = FeatureExtractor()
        self.encoder = Encoder(input_size=self._get_encoder_input_size(height),
                               hidden_size=enc_hidden_size,
                               num_layers=enc_num_layers)

    def forward(self, x):
        x = self.fe(x)
        h = self.encoder(x)

        return h

    @torch.no_grad()
    def _get_encoder_input_size(self, height):
        x = torch.rand(1, 1, height, 224)
        y = self.fe(x)

        return y.shape[1]


# dont forget about FPN
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = timm.create_model('resnet18',
                                    num_classes=0,
                                    global_pool='',
                                    in_chans=1)

    def forward(self, x):
        x = self.fe(x)  # bs, 512, 4, fe_width
        return torch.mean(x, dim=2)  # bs, 512, fe_width


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=512,
                 num_layers=3,
                 bidirectional=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional)
        self.conv = nn.Conv1d(input_size, input_size, (1,))

    def forward(self, x):
        seq = x.permute(2, 0, 1)  # seq_len(width), bs, input_size(channels)

        bs = x.size(0)
        hidden_states = []
        h, c = self.init_hidden(bs, x.dtype, x.device)
        for x_t in seq:
            y, (h, c) = self.lstm(x_t.unsqueeze(0), (h, c))
            hidden_states.append(h.view(self.num_layers,
                                        self.num_directions,
                                        bs,
                                        self.hidden_size)[-1, -1])

        y = torch.stack(hidden_states, dim=0)  # seq_len, bs, hidden_size
        y = y.permute(1, 2, 0)  # bs, seq_len(width), hidden_size

        return self.conv(y)  # bs, channels, width

    def init_hidden(self, bs, dtype, device):
        num_directions = 2 if self.bidirectional else 1
        h_zeros = torch.zeros(self.num_layers * num_directions,
                              bs, self.hidden_size,
                              dtype=dtype, device=device)
        c_zeros = torch.zeros(self.num_layers * num_directions,
                              bs, self.hidden_size,
                              dtype=dtype, device=device)

        return h_zeros, c_zeros
