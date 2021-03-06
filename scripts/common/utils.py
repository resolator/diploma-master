#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import fastwer
from .baseline_net import BaselineNet
from .seq2seq_model import Seq2seqModel
from .seq2seq_light_model import Seq2seqLightModel
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

    common_args = {
        'c2i': c2i,
        'i2c': i2c,
        'dec_hs': getattr(args, 'dec_hs',  256),
        'backbone_out': getattr(args, 'backbone_out', 256),
        'fe_dropout': getattr(args, 'fe_dropout', 0.15),
        'dec_dropout': getattr(args, 'dec_dropout', 0.2),
        'dec_n_layers': getattr(args, 'dec_layers', 1),
        'rnn_dropout': getattr(args, 'rnn_dropout', 0),
        'rnn_type': getattr(args, 'rnn_type', 'lstm')
    }
    if args.model_type == 'baseline':
        model = BaselineNet(**common_args)
    elif args.model_type == 'seq2seq':
        model = Seq2seqModel(
            text_max_len=args.text_max_len,
            enc_hs=args.enc_hs,
            attn_sz=args.attn_size,
            emb_size=args.emb_size,
            enc_n_layers=args.enc_layers,
            pe=getattr(args, 'pos_encoding', False),
            teacher_rate=args.teacher_rate,
            **common_args
        )
    elif args.model_type == 'seq2seq_light':
        model = Seq2seqLightModel(
            text_max_len=args.text_max_len,
            attn_sz=args.attn_size,
            emb_size=args.emb_size,
            pe=getattr(args, 'pos_encoding', False),
            teacher_rate=args.teacher_rate,
            backbone=getattr(args, 'backbone', 'conv_net6'),
            expand_h=getattr(args, 'expand_h', False),
            k=getattr(args, 'dsc_k', 1),
            gates=getattr(args, 'gates', 0),
            gate_dropout=getattr(args, 'gate_dropout', 0.4),
            gate_width=getattr(args, 'gate_width', 9),
            **common_args
        )
    else:
        raise AssertionError(
            'model type must be in [baseline, seq2seq, seq2seq_light]'
        )

    return model


def calc_cer(gt, pd, gt_lens=None):
    if gt_lens is not None:
        gt = [x[:y] for x, y in zip(gt, gt_lens)]
        pd = [x.replace('_', '')[:y] for x, y in zip(pd, gt_lens)]

    return fastwer.score(pd, gt, char_level=True)


def calc_params(params, check_requires_grad=False):
    if check_requires_grad:
        return sum(p.numel() for p in params if p.requires_grad)
    else:
        return sum(p.numel() for p in params)


def print_model_params(params, trainable=True):
    print('=' * 32)
    print('All parameters:       ', end='')
    print('{:,}'.format(calc_params(params)))

    if trainable:
        print('Trainable parameters: ', end='')
        print('{:,}'.format(calc_params(params, True)))
        print('=' * 32)
