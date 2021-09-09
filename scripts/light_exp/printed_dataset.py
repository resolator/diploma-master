#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Template doc"""
import cv2
import torch

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torch.nn import functional as F


class PrintedDataset(Dataset):
    def __init__(self, ds_dir, c2i, i2c, text_max_len=62):
        ds_dir = Path(ds_dir)
        self.images_paths = sorted(ds_dir.glob('*.png'))
        self.texts_paths = sorted(ds_dir.glob('*.txt'))

        assert len(self.texts_paths) == len(self.images_paths)

        self.c2i = c2i
        self.i2c = i2c
        self.max_len = text_max_len + 1

        self.texts = []
        self.lens = []
        for text_path in self.texts_paths:
            with open(text_path, 'r') as f:
                line_text = f.read()

            # convert text to tensor with padding
            line_tensor = torch.tensor([self.c2i[x] for x in line_text],
                                       dtype=torch.int64)
            line_len = len(line_tensor)
            diff = self.max_len - line_len
            self.texts.append(F.pad(
                line_tensor, (0, diff), 'constant', self.c2i['é']
            ))
            self.lens.append(line_len + 1)

    def __len__(self):
        return len(self.texts_paths)

    def __getitem__(self, idx):
        text = self.texts[idx]
        length = self.lens[idx]

        img = cv2.imread(str(self.images_paths[idx]), cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255

        return img, text, length

    def show_dataset(self, n_samples):
        assert n_samples < len(self), \
            f'dataset size is {len(self)}, however ' \
            f'n_samples for visualization is {n_samples}.'

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        for i in range(n_samples):
            img, text, lens = self[i]
            img = (img.squeeze() * 255).numpy().astype(np.uint8)

            print(self.tensor2text(text[:lens]))

            cv2.imshow('img', img)
            cv2.waitKey()

        cv2.destroyWindow('img')

    def tensor2text(self, tensor):
        strings = []

        if isinstance(tensor, list):
            for sample in tensor:
                strings.append(''.join([self.i2c[x] for x in sample]))

        else:
            if not isinstance(tensor, np.ndarray):
                tensor = tensor.detach().cpu().numpy()

            if len(tensor.shape) == 1:
                return ''.join([self.i2c[x] for x in tensor])
            else:
                for sample in tensor:
                    strings.append(''.join([self.i2c[x] for x in sample]))

        return strings
