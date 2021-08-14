#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
from string import digits, ascii_letters


def build_alphabet():
    symbs = ['ƀ', 'ś', 'é', ' ', '!', '"', '#', '&', '\'',
             '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']
    i2c = symbs + list(digits) + list(ascii_letters)
    c2i = {c: idx for idx, c in enumerate(i2c)}

    return c2i, i2c
