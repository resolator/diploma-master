#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Evaluation script."""
import torch
import argparse

from tqdm import tqdm
from pathlib import Path
from utils import create_model, calc_cer
from torch.utils.data import DataLoader
from printed_dataset import PrintedDataset


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ds-dir', type=Path, required=True,
                        help='Path to dataset directort.')
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path to the trained model.')
    parser.add_argument('--bs', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    # model loading
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    ckpt = torch.load(args.model_path, map_location=device)

    i2c = ckpt['i2c']
    c2i = {c: idx for idx, c in enumerate(i2c)}

    model = create_model(c2i, i2c).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # create dataset
    text_max_len = 62
    ds = PrintedDataset(args.ds_dir, c2i, i2c, text_max_len=text_max_len)
    dl = DataLoader(ds, args.bs, num_workers=4)

    # evaluation
    cer = 0
    loader_size = len(dl)
    for img, text, lens in tqdm(dl, desc='Evaluating'):
        img, text, lens = img.to(device), text.to(device), lens.to(device)
        logits, log_probs = model(img)

        # cer
        pd_beam, pd_lens = model.decode(log_probs)

        pd_beam = ds.tensor2text(pd_beam)
        gt_text = ds.tensor2text(text)

        gt_lens = lens.detach().cpu().numpy()
        cer = calc_cer(gt_text, pd_beam, gt_lens)
        if cer > 0:
            print(cer)
        cer += cer / loader_size

    print(('Mean CER: %.3f' % cer) + '%')


if __name__ == '__main__':
    main()
