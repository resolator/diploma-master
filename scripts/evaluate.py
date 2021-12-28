#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Evaluation script."""
import torch
import argparse

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from common.utils import create_model, calc_cer
from torch.utils.data import DataLoader
from iam_ds.iam_dataset import IAMDataset


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to root dir with images (ex. iam/lines/).')
    parser.add_argument('--mkp-dir', type=Path, required=True,
                        help='Path to dir with xml files (ex. iam/xml).')
    parser.add_argument('--split', type=Path, required=True,
                        help='Path to split file. Can be generated '
                             'using scripts/utils/gen_split_file.py.')
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path to the trained model.')
    parser.add_argument('--bs', type=int, default=64,
                        help='Batch size.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    # model loading
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print('Checkpoint loading')
    ckpt = torch.load(args.model_path, map_location=device)

    i2c = ckpt['i2c']
    c2i = {c: idx for idx, c in enumerate(i2c)}

    model = create_model(c2i, i2c).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print('This model metrics:')
    pprint(ckpt['metrics'])

    print('Creating dataset')
    text_max_len = 98
    ds = IAMDataset(args.images_dir, args.mkp_dir, args.split,
                    i2c, ckpt['args'].height, text_max_len)
    dl = DataLoader(ds, args.bs, num_workers=4, collate_fn=ds.collate_fn)

    # evaluation
    final_cer = 0.0
    loader_size = len(dl)
    for img, text, lens in tqdm(dl, desc='Evaluating'):
        img, text, lens = img.to(device), text.to(device), lens.to(device)
        _, log_probs = model(img)

        # cer
        gt_text = dl.dataset.tensor2text(text)
        pd_beam, _ = model.decode(log_probs)
        pd_beam = dl.dataset.tensor2text(pd_beam)

        gt_lens = lens.detach().cpu().numpy()
        cer = calc_cer(gt_text, pd_beam, gt_lens)
        final_cer += cer / loader_size

    print(('Mean CER: %.3f' % final_cer) + '%')


if __name__ == '__main__':
    main()
