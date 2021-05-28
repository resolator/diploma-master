#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""PyTorch wrapper for IAM dataset."""
import cv2
import torch

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class IAMDataset(Dataset):
    def __init__(self,
                 images_dir,
                 markup_filepath,
                 split_filepath,
                 height=119):
        self.height = height
        self.markup = self._read_markup_file(markup_filepath)

        # read images from split only
        with open(split_filepath, 'r') as f:
            names = [x[:-1] for x in f.readlines()]

        self.imgs_paths = [x for x in Path(images_dir).rglob('*.png')
                           if x.stem in names]

    def __len__(self):
        return len(self.imgs_paths)

    @staticmethod
    def _resize(img, height):
        """Resize image with saving aspect ratio."""
        if img.shape[0] == height:
            return img

        # calculate ratio
        ratio = height / img.shape[0]

        return cv2.resize(img, None, fx=ratio, fy=ratio,
                          interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = self._resize(img, self.height)

        img = (torch.tensor(img).unsqueeze(0) / 255.0 - 0.5) * 2

        text = self.markup[img_path.stem]

        return img, text

    @staticmethod
    def _read_markup_file(markup_filepath):
        # read markup file lines
        with open(markup_filepath, 'r') as f:
            lines = f.readlines()

        markup = {}
        for line in lines:
            # skip comments
            if line[0] == '#':
                continue

            # make a record
            line = line.split(' ')
            name = line[0]
            text = line[-1].split('|')[:-1]
            markup[name] = text

        return markup

    def show_dataset(self, n_samples):
        assert n_samples < len(self), \
            f'dataset size is {len(self)}, however ' \
            f'n_samples for visualization is {n_samples}.'

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        for i in range(n_samples):
            img, text = self[i]
            img = ((img.squeeze() / 2 + 0.5) * 255).numpy().astype(np.uint8)
            print(' '.join(text))
            cv2.imshow('img', img)
            cv2.waitKey()

        cv2.destroyWindow('img')
