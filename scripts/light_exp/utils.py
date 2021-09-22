#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import fastwer
from baseline_net import BaselineNet
from seq2seq_model import Seq2seqModel
from string import digits, ascii_letters


def build_alphabet(light=True, with_ctc_blank=True, with_sos=False):
    i2c = ['_'] if with_ctc_blank else []
    i2c = i2c + ['ś'] if with_sos else i2c

    if light:
        i2c = i2c + [' ', 'é', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k',
                     'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'x',
                     'y', 'z']
    else:
        symbs = ['é', ' ', '!', '"', '#', '&', '\'', '(', ')', '*', '+',
                 ',', '-', '.', '/', ':', ';', '?']
        i2c = i2c + symbs + list(digits) + list(ascii_letters)

    c2i = {c: idx for idx, c in enumerate(i2c)}

    return c2i, i2c


def create_model(c2i, i2c, model_type='ctc', text_max_len=62):
    """Wrapper for creating different models."""

    if model_type == 'ctc':
        model = BaselineNet(c2i, i2c)
    elif model_type == 'seq2seq':
        model = Seq2seqModel(i2c, text_max_len)
    else:
        raise AssertionError('model type must be "ctc" or "seq2seq"')

    return model


def calc_cer(gt, pd, gt_lens=None):
    if gt_lens is not None:
        gt = [x[:y] for x, y in zip(gt, gt_lens)]
        pd = [x.replace('_', '')[:y] for x, y in zip(pd, gt_lens)]

    return fastwer.score(pd, gt, char_level=True)
