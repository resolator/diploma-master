#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""PyTorch wrapper for IAM dataset."""
import cv2

from pathlib import Path
from torch.utils.data import Dataset


class IAMDataset(Dataset):
    def __init__(self, images_dir, markup_filepath, split_filepath):
        self.markup = self._read_markup_file(markup_filepath)

        # read images from split only
        with open(split_filepath, 'r') as f:
            names = [x[:-1] for x in f.readlines()]

        self.imgs_paths = [x for x in Path(images_dir).rglob('*.png')
                           if x.stem in names]

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = cv2.imread(str(img_path))
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
            text = line[-1].split('|')
            markup[name] = text

        return markup

    def show_dataset(self, n_samples):
        assert n_samples < len(self), \
            f'dataset size is {len(self)}, however ' \
            f'n_samples for visualization is {n_samples}.'

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        for i in range(n_samples):
            img, text = self[i]
            print(' '.join(text))
            cv2.imshow('img', img)
            cv2.waitKey()
