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
                        help='Save errors to this dir.'
                             'In this case bs will be 1.')
    parser.add_argument('--save-atts', action='store_true',
                        help='Draw attention on the image if possible and save '
                             'it to attn dir at model\'s dir level.')

    args = parser.parse_args()

    if args.save_errors_to is not None:
        print('--save-errors-to passed, bs is set to 1.')
        args.bs = 1
        args.save_errors_to.mkdir(exist_ok=True, parents=True)

    return args


def draw_attention(img, atts, pd_text, save_to):
    if len(atts.shape) == 2:
        atts.unsqueeze(-2)  # added H dim

    # addet C dim (AS, H, W, C)
    img = img.detach().cpu().squeeze(0).unsqueeze(-1).numpy()
    atts = atts.unsqueeze(-1).detach().cpu().numpy().astype(np.float32)
    for i in range(atts.shape[0]):
        atts[i] = atts[i] / atts[i].max()


    # cutting tralling <eos>
    pd_text = pd_text[0]
    idx = len(pd_text)
    for i in range(len(pd_text) - 1, 0, -1):
        if pd_text[i] != 'é':
            break
        idx = i + 1

    pd_text = pd_text[:idx]
    atts = atts[:idx]

    # draw and save each char
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for idx, (att, char) in enumerate(zip(atts, pd_text)):
        resized_att = cv2.resize(att, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        attended_img = color_img.copy()
        attended_img[:, :, 0] -= resized_att / 2
        attended_img[:, :, 1] -= resized_att / 2
        attended_img[:, :, 2] += resized_att / 2
        attended_img = (np.clip(attended_img, 0, 1) * 255).astype(np.uint8)

        img_name = str(idx) + '_attended_' + char + '.png'
        cv2.imwrite(str(save_to.joinpath(img_name)), attended_img)


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
                    return_path=True,
                    correction=getattr(ckpt['args'], 'correction', True))
    dl = DataLoader(ds, args.bs, num_workers=4, collate_fn=ds.collate_fn)

    print('Model creation')
    model = create_model(c2i, i2c, ckpt['args']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model_type = ckpt['args'].model_type

    print('This model metrics:')
    pprint(ckpt['metrics'])

    # evaluation
    final_cer, final_loss = 0.0, 0.0
    loader_size = len(dl)
    errors = {}
    for idx, (img, text, lens, paths) in enumerate(tqdm(dl, desc='Evaluating')):
        img, text, lens = img.to(device), text.to(device), lens.to(device)

        if model_type == 'baseline':
            _, log_probs = model(img)
            pd_text, _ = model.decode(log_probs)
            pd_text = ds.tensor2text(pd_text)
            atts = None
        else:
            log_probs, preds, atts = model(img)
            pd_text = ds.tensor2text(preds)

        # cer
        gt_text = ds.tensor2text(text)
        gt_lens = lens.detach().cpu().numpy()
        cer = calc_cer(gt_text, pd_text, gt_lens)
        final_cer += cer / loader_size

        # loss
        final_loss += model.calc_loss(log_probs,
                                      text,
                                      lens).item() / loader_size

        # attention
        if args.save_atts and (atts is not None):
            atts_dir = args.model_path.parent.parent / 'atts'
            atts_dir.mkdir(exist_ok=True)

            for im, att, img_path in zip(img, atts, paths):
                img_dir = atts_dir / img_path.stem
                img_dir.mkdir(exist_ok=True)
                draw_attention(im, att, pd_text, img_dir)

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
    print(('Mean loss: %.3f' % final_loss) + '%')


if __name__ == '__main__':
    main()
