#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""PyTorch wrapper for IAM dataset."""
import cv2
import torch

import numpy as np
import albumentations as albu
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from xml.etree import ElementTree as ET

try:
    from transforms import SkewCorrection, SlantCorrection, ContrastNormalization
except ModuleNotFoundError:
    from .transforms import SkewCorrection, SlantCorrection, ContrastNormalization


class IAMDataset(Dataset):
    """PyTorch wrapper for IAM dataset."""
    def __init__(self,
                 images_dir,
                 markup_dir,
                 split_filepath,
                 i2c,
                 height=64,
                 width=None,
                 max_len=300,
                 augment=False,
                 correction=True,
                 return_path=False,
                 load_to_ram=False):
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
        width : int
            Pad all image to this width. Must greater or equal to the maximal
            width in the dataset.
        max_len : int
            Maximal length for text sequence.
        augment : bool
            Augment images.
        correction : bool
            Correct slant, skew and contrast.
        return_path : bool
            Return the path of an image in __getitem__.
        load_to_ram : bool
            Load images to RAM.

        """
        self.return_path = return_path
        self.i2c = i2c
        self.c2i = {c: idx for idx, c in enumerate(self.i2c)}

        self.height = height
        self.width = width
        self.max_len = max_len
        self.augment = augment
        self.load_to_ram = load_to_ram

        # read images from split only
        with open(split_filepath, 'r') as f:
            names = [x.replace('\n', '') for x in f.readlines()]

        self.imgs_paths = [x for x in Path(images_dir).rglob('*.png')
                           if x.stem in names]

        self.markup, self.lens = self._read_markup(markup_dir)

        corr_transform = albu.Compose([
            SkewCorrection(p=1),
            SlantCorrection(p=1),
            ContrastNormalization(p=1)
        ]) if correction else albu.Compose([])

        if self.augment:
            self.transform = albu.Compose([
                albu.CropAndPad([[12, 20], [24, 40], [12, 20], [24, 40]],
                                pad_mode=cv2.BORDER_REPLICATE,
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
                corr_transform
            ])
        else:
            self.transform = albu.Compose([
                albu.CropAndPad([16, 32, 16, 32],
                                pad_mode=cv2.BORDER_CONSTANT,
                                pad_cval=255,
                                keep_size=False),
                albu.SmallestMaxSize(self.height),
                corr_transform
            ])

        self.images = None
        if self.load_to_ram:
            self.images = []
            for img_path in tqdm(self.imgs_paths, desc='Loading images to RAM'):
                self.images.append(self.get_image(img_path))

    def get_image(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = self.transform(image=img)['image']

        if self.width is not None:
            if self.width < img.shape[-1]:
                print('WARNING: padding width is lower than actual image width')
            else:
                img = cv2.copyMakeBorder(img, 0, 0, 0,
                                         self.width - img.shape[-1],
                                         cv2.BORDER_REPLICATE)

        return (torch.tensor(img).unsqueeze(0) / 255.0)

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        if self.images is not None:
            img = self.images[idx]
        else:
            img = self.get_image(img_path)

        text = self.markup[idx]
        length = self.lens[idx]

        if self.return_path:
            return img, text, length, img_path

        return img, text, length

    def _read_markup(self, markup_dir):
        # read markup from xml files
        markup_dict = {}
        for xml_path in Path(markup_dir).rglob('*.xml'):
            lines = ET.parse(xml_path).getroot()[1]
            for line in lines:
                line_id = line.attrib['id']
                line_text = line.attrib['text']
                line_text = IAMDataset.html_decode(line_text)

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
    def html_decode(line):
        """Decode HTML tags from markup text."""
        htmlCodes = (("'", '&#39;'),
                     ('"', '&quot;'),
                     ('&', '&amp;'))

        for code in htmlCodes:
            line = line.replace(code[1], code[0])

        return line

    @staticmethod
    def collate_fn(batch):
        texts = torch.stack([x[1] for x in batch])
        lens = torch.stack([x[2] for x in batch])

        paths = None
        if len(batch[0]) == 4:
            paths = [x[3] for x in batch]

        images = [x[0] for x in batch]
        widths = [x.size(2) for x in images]
        max_width = torch.max(torch.tensor(widths)).long()
        images = [F.pad(x, (0, max_width - x.size(2)), 'replicate')
                  for x in images]

        if paths is not None:
            return torch.stack(images), texts, lens, paths

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
