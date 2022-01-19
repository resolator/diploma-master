#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import fastwer
from .baseline_net import BaselineNet
from .seq2seq_model import Seq2seqModel
from .seg_attn_model import SegAttnModel
from string import digits, ascii_letters


def build_alphabet(light=True, with_ctc_blank=True, with_sos=False):
    i2c = ['_'] if with_ctc_blank else []
    i2c = i2c + ['ś'] if with_sos else i2c

    if light:
        i2c = i2c + [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k',
                     'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'x',
                     'y', 'z', 'é']
    else:
        symbs = [' ', '!', '"', '#', '&', '\'', '(', ')', '*', '+',
                 ',', '-', '.', '/', ':', ';', '?', 'é']
        i2c = i2c + symbs + list(digits) + list(ascii_letters)

    c2i = {c: idx for idx, c in enumerate(i2c)}

    return c2i, i2c


def create_model(c2i, i2c, args):
    """Wrapper for creating different models."""

    if args.model_type == 'baseline':
        model = BaselineNet(c2i, i2c, args.n_layers)
    elif args.model_type == 'seq2seq':
        model = Seq2seqModel(c2i=c2i,
                             i2c=i2c,
                             text_max_len=args.text_max_len,
                             enc_hs=args.enc_hs,
                             emb_size=args.emb_size,
                             enc_n_layers=args.enc_layers,
                             pe=args.pos_encoding,
                             teacher_rate=args.teacher_rate)
    elif args.model_type == 'seg_attn':
        model = SegAttnModel(c2i=c2i,
                             i2c=i2c,
                             text_max_len=args.text_max_len,
                             backbone_out=256,
                             dec_dropout=args.dec_dropout,
                             teacher_rate=args.teacher_rate,
                             decoder_type=args.decoder_type,
                             fe_dropout=args.fe_dropout,
                             emb_size=args.emb_size)
    else:
        raise AssertionError(
            'model type must be in [baseline, seq2seq, seg_attn]'
        )

    return model


def calc_cer(gt, pd, gt_lens=None):
    if gt_lens is not None:
        gt = [x[:y] for x, y in zip(gt, gt_lens)]
        pd = [x.replace('_', '')[:y] for x, y in zip(pd, gt_lens)]

    return fastwer.score(pd, gt, char_level=True)
