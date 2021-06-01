#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""PyTorch wrapper for IAM dataset."""
import cv2
import torch

import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
from xml.etree import ElementTree as ET


class IAMDataset(Dataset):
    """PyTorch wrapper for IAM dataset."""
    def __init__(self,
                 images_dir,
                 markup_dir,
                 split_filepath,
                 c2i,
                 height=119,
                 max_len=127):
        """Initialize IAM dataset.

        Parameters
        ----------
        images_dir : str or pathlib.Path
            Path to root dir with images (ex. iam/lines/).
        markup_dir : str or pathlib.Path
            Path to markup dir with xml files (ex. iam/xml).
        split_filepath : str or pathlib.Path
            Path to split file.
        c2i : dict
            Model's converter from character to index.
        height : int
            Target height for images.
        max_len : int
            Maximal length for text sequence.

        """
        self.height = height
        self.c2i = c2i
        self.i2c = [x[0] for x in sorted(self.c2i.items(), key=lambda x: x[1])]

        self.max_len = max_len + 1  # +1 for <eos>

        # read images from split only
        with open(split_filepath, 'r') as f:
            names = [x[:-1] for x in f.readlines()]

        self.imgs_paths = [x for x in Path(images_dir).rglob('*.png')
                           if x.stem in names]

        self.markup, self.lens = self._read_markup(markup_dir)

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
                                    self.c2i['<eos>'])

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
        if not numpy:
            tensor = tensor.detach().cpu().numpy()

        if len(tensor.shape) == 1:
            return ''.join([self.i2c[x] for x in tensor])
        else:
            strings = []
            for sample in tensor:
                strings.append(''.join([self.i2c[x] for x in sample]))

            return strings

    def show_dataset(self, n_samples):
        assert n_samples < len(self), \
            f'dataset size is {len(self)}, however ' \
            f'n_samples for visualization is {n_samples}.'

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        for i in range(n_samples):
            img, text, lens = self.collate_fn([self[i]])
            img = ((img.squeeze() / 2 + 0.5) * 255).numpy().astype(np.uint8)

            print(self.tensor2text(text[0][:lens[0]]))

            cv2.imshow('img', img)
            cv2.waitKey()

        cv2.destroyWindow('img')
