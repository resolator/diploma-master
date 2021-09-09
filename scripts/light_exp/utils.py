#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import fastwer
from baseline_net import BaselineNet


def build_alphabet():
    i2c = ['_', ' ', 'é', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'm', 'n',
           'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'x', 'y', 'z']

    c2i = {c: idx for idx, c in enumerate(i2c)}

    return c2i, i2c


def create_model(c2i, i2c):
    """Wrapper for creating different models."""

    model = BaselineNet(c2i, i2c)
    return model


def calc_cer(gt, pd, gt_lens=None):
    if gt_lens is not None:
        gt = [x[:y] for x, y in zip(gt, gt_lens)]
        pd = [x.replace('_', '')[:y] for x, y in zip(pd, gt_lens)]

    return fastwer.score(pd, gt, char_level=True)
