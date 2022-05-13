#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Evaluation script."""
import cv2
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from shutil import copyfile
from common.utils import create_model, calc_cer
from torch.utils.data import DataLoader
from iam_ds.iam_dataset import IAMDataset


def get_args():
    """Arguments parser and checker."""
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
    parser.add_argument('--save-errors-to', type=Path,
                        help='Save errors to this file.'
                             'In this case bs will be 1.')

    args = parser.parse_args()

    if args.save_errors_to is not None:
        print('--save-errors-to passed, bs is set to 1.')
        args.bs = 1
        args.save_errors_to.mkdir(exist_ok=True, parents=True)

    return args


def main():
    """Application entry point."""
    args = get_args()

    print('Checkpoint loading')
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    ckpt = torch.load(args.model_path, map_location=device)

    i2c = ckpt['i2c']
    c2i = {c: idx for idx, c in enumerate(i2c)}

    print('Creating dataset')
    text_max_len = ckpt['args'].text_max_len
    ds = IAMDataset(images_dir=args.images_dir,
                    markup_dir=args.mkp_dir,
                    split_filepath=args.split,
                    i2c=i2c,
                    height=ckpt['args'].height,
                    width=getattr(ckpt['args'], 'img_max_width'),
                    max_len=text_max_len,
                    augment=False,
                    return_path=True)
    dl = DataLoader(ds, args.bs, num_workers=4, collate_fn=ds.collate_fn,
                    drop_last=True)

    print('Model creation')
    model = create_model(c2i, i2c, ckpt['args']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model_type = ckpt['args'].model_type

    print('This model metrics:')
    pprint(ckpt['metrics'])

    # evaluation
    final_cer = 0.0
    loader_size = len(dl)
    errors = {}
    for idx, (img, text, lens, paths) in enumerate(tqdm(dl, desc='Evaluating')):
        img, text, lens = img.to(device), text.to(device), lens.to(device)

        if model_type == 'baseline':
            _, log_probs = model(img)
            pd_text, _ = model.decode(log_probs)
            pd_text = ds.tensor2text(pd_text)
        else:
            log_probs, preds, _ = model(img)
            pd_text = ds.tensor2text(preds)

        # cer
        gt_text = ds.tensor2text(text)
        gt_lens = lens.detach().cpu().numpy()
        cer = calc_cer(gt_text, pd_text, gt_lens)
        final_cer += cer / loader_size

        # save errors
        if (args.save_errors_to is not None) and cer > 0:
            img_name = str(idx) + '.png'
            errors[img_name] = ({'name': paths[0].name,
                                 'gt': gt_text[0][:gt_lens[0]],
                                 'pd': pd_text[0][:gt_lens[0]]})

            img_path = args.save_errors_to.joinpath(img_name)
            copyfile(paths[0], img_path)

            img_path_aug = args.save_errors_to.joinpath(
                img_path.stem + '_aug.' + img_path.suffix
            )
            img = (img.squeeze() * 255).cpu().numpy().astype(np.uint8)
            cv2.imwrite(str(img_path_aug), img)

    if args.save_errors_to is not None:
        with open(args.save_errors_to.joinpath('errors.json'), 'w') as f:
            json.dump(errors, fp=f, indent=4)

    print(('Mean CER: %.3f' % final_cer) + '%')


if __name__ == '__main__':
    main()
