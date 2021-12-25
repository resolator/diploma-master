#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""PyTorch wrapper for IAM dataset."""
import cv2
import torch

import numpy as np
import albumentations as albu
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
from xml.etree import ElementTree as ET

import transforms as my_t


class IAMDataset(Dataset):
    """PyTorch wrapper for IAM dataset."""
    def __init__(self,
                 images_dir,
                 markup_dir,
                 split_filepath,
                 i2c,
                 height=64,
                 max_len=300,
                 augment=False):
        """Initialize IAM dataset.

        Parameters
        ----------
        images_dir : str or pathlib.Path
            Path to root dir with images (ex. iam/lines/).
        markup_dir : str or pathlib.Path
            Path to markup dir with xml files (ex. iam/xml).
        split_filepath : str or pathlib.Path
            Path to split file.
        i2c : array-like
            Model's converter from index to character.
        height : int
            Target height for images.
        max_len : int
            Maximal length for text sequence.
        augment : bool
            Auggment images.

        """
        self.i2c = i2c
        self.c2i = {c: idx for idx, c in enumerate(self.i2c)}

        self.height = height
        self.max_len = max_len
        self.augment = augment

        # read images from split only
        with open(split_filepath, 'r') as f:
            names = [x.replace('\n', '') for x in f.readlines()]

        self.imgs_paths = [x for x in Path(images_dir).rglob('*.png')
                           if x.stem in names]

        self.markup, self.lens = self._read_markup(markup_dir)

        if self.augment:
            self.transform = albu.Compose([
                albu.CropAndPad([[12, 20], [24, 40], [12, 20], [24, 40]],
                                pad_cval=255,
                                keep_size=False),
                albu.GaussianBlur(3, 3, p=0.4),
                albu.Affine(scale=[0.9, 1.0],
                            shear={'x': [-9, 9]},
                            cval=255,
                            fit_output=True,
                            p=0.7),
                albu.SmallestMaxSize(self.height),
                albu.CLAHE(),
                albu.Emboss(),
                albu.RandomBrightnessContrast(),
                my_t.SkewCorrection(),
                my_t.SlantCorrection(),
                my_t.ContrastNormalization()
            ])
        else:
            self.transform = albu.Compose([
                albu.CropAndPad([16, 32, 16, 32],
                                pad_cval=255,
                                keep_size=False,
                                always_apply=True),
                albu.SmallestMaxSize(self.height),
                my_t.SkewCorrection(),
                my_t.SlantCorrection(),
                my_t.ContrastNormalization()
            ])

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        img = self.transform(image=img)['image']
        img = (torch.tensor(img).unsqueeze(0) / 255.0)

        text = self.markup[idx]
        length = self.lens[idx]

        return img, text, length

    def _read_markup(self, markup_dir):
        # read markup from xml files
        markup_dict = {}
        for xml_path in Path(markup_dir).rglob('*.xml'):
            lines = ET.parse(xml_path).getroot()[1]
            for line in lines:
                line_id = line.attrib['id']
                line_text = line.attrib['text']

                # convert text to tensor with padding
                line_tensor = torch.tensor([self.c2i[x] for x in line_text],
                                           dtype=torch.int64)
                line_len = len(line_tensor)
                diff = self.max_len - line_len
                line_tensor = F.pad(line_tensor,
                                    (0, diff),
                                    'constant',
                                    self.c2i['é'])

                # +1 for <eos>
                markup_dict[line_id] = [line_tensor,
                                        torch.tensor(line_len + 1)]

        # reorder markup according to images order
        markup = []
        lens = []
        for img_path in self.imgs_paths:
            markup.append(markup_dict[img_path.stem][0])
            lens.append(markup_dict[img_path.stem][1])

        return markup, lens

    @staticmethod
    def collate_fn(batch, pad_value=1):
        texts = torch.stack([x[1] for x in batch])
        lens = torch.stack([x[2] for x in batch])

        images = [x[0] for x in batch]
        widths = [x.size(2) for x in images]
        max_width = torch.max(torch.tensor(widths)).long()
        images = [F.pad(x, (0, max_width - x.size(2)), 'constant', pad_value)
                  for x in images]

        return torch.stack(images), texts, lens

    def tensor2text(self, tensor, numpy=False):
        strings = []

        if isinstance(tensor, list):
            for sample in tensor:
                strings.append(''.join([self.i2c[x] for x in sample]))

        else:
            if not numpy:
                tensor = tensor.detach().cpu().numpy()

            if len(tensor.shape) == 1:
                return ''.join([self.i2c[x] for x in tensor])
            else:
                for sample in tensor:
                    strings.append(''.join([self.i2c[x] for x in sample]))

        return strings

    def show_dataset(self, n_samples):
        assert n_samples <= len(self), \
            f'dataset size is {len(self)}, however ' \
            f'n_samples for visualization is {n_samples}.'

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        try:
            for i in range(n_samples):
                img, text, lens = self.collate_fn([self[i]])
                img = (img.squeeze() * 255).numpy().astype(np.uint8)

                print('\nImage shape:', img.shape)
                print(self.tensor2text(text[0][:lens[0]]))

                cv2.imshow('img', img)
                cv2.waitKey()
        finally:
            cv2.destroyWindow('img')