#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Training launcher."""
import sys
import torch
import configargparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from iam_dataset import IAMDataset

sys.path.append(str(Path(sys.path[0]).parent))
from common.utils import build_alphabet, create_model, calc_cer


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', is_config_file=True,
                        help='Path to config file.')

    # dataset
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Path to root dir with images (ex. iam/lines/).')
    parser.add_argument('--mkp-dir', type=Path, required=True,
                        help='Path to dir with xml files (ex. iam/xml).')
    parser.add_argument('--train-split', type=Path, required=True,
                        help='Path to train split file. Can be generated '
                             'using scripts/utils/gen_split_file.py.')
    parser.add_argument('--valid-split', type=Path, required=True,
                        help='Path to validation split file. Can be generated '
                             'using scripts/utils/gen_split_file.py.')
    parser.add_argument('--ckpt-path', type=Path,
                        help='Path to saved model to load for training.')

    # training
    parser.add_argument('--epochs', type=int, default=-1,
                        help='Number of epochs to train. '
                             'Pass -1 for infinite training.')
    parser.add_argument('--bs', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of data loader workers.')
    parser.add_argument('--augment', action='store_true',
                        help='Augment images.')

    # model
    parser.add_argument('--height', type=int, default=64,
                        help='Input image height. Will resize to this value.')
    parser.add_argument('--pos-encoding', action='store_true',
                        help='Add positional encoding before decoder.')
    parser.add_argument('--enc-hs', type=int, default=128,
                        help='(bidirectional) Encoder hidden size.')
    parser.add_argument('--emb-size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--dec-hs', type=int, default=128,
                        help='Decoder hidden size.')
    parser.add_argument('--teacher-rate', type=float, default=1.0,
                        help='Teacher rate for decoder training input.')
    parser.add_argument('--enc-layers', type=int, default=1,
                        help='Encoder RNN layers.')
    parser.add_argument('--dec-layers', type=int, default=1,
                        help='Decoder RNN layers.')

    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def save_model(model, optim, args, ep, metrics, best_metrics, models_dir):
    """Save best models and update best metrics."""
    save_data = {'model': model.state_dict(),
                 'optim': optim.state_dict(),
                 'args': args,
                 'epoch': ep,
                 'metrics': metrics,
                 'best_metrics': best_metrics,
                 'i2c': model.i2c}

    for m in metrics.keys():
        for stage in ['train', 'valid']:
            if metrics[m][stage] < best_metrics[m][stage]:
                model_path = models_dir.joinpath(stage + '-' + m + '.pth')
                torch.save(save_data, model_path)
                best_metrics[m][stage] = metrics[m][stage]

                print(f'Saved {stage} {m}')


def load_model(ckpt_path, model, optim, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optim.load_state_dict(ckpt['optim'])
    best_metrics = ckpt['best_metrics']
    
    return model, optim, best_metrics


def epoch_step(model, loaders, device, optim, writer, epoch):
    metrics = {'cer': {'train': 0.0, 'valid': 0.0},
               'loss': {'train': 0.0, 'valid': 0.0}}

    for stage in ['train', 'valid']:
        is_train = stage == 'train'
        model.train() if is_train else model.eval()
        loader_size = len(loaders[stage])

        torch.set_grad_enabled(is_train)
        img_count = 0
        for i, (img, text, lens) in enumerate(tqdm(loaders[stage], desc=stage)):
            if is_train:
                optim.zero_grad()

            # forward
            img, text, lens = img.to(device), text.to(device), lens.to(device)
            logs_probs, preds, atts = model(img)

            # loss
            loss = model.calc_loss(logs_probs, text, lens)
            metrics['loss'][stage] += loss.item() / loader_size

            # cer
            gt_text = loaders[stage].dataset.tensor2text(text)
            gt_lens = lens.detach().cpu().numpy()

            pd_text = loaders[stage].dataset.tensor2text(preds)
            cer = calc_cer(gt_text, pd_text, gt_lens)
            metrics['cer'][stage] += cer / loader_size

            # dump attention
            if i % 15 == 0:
                att = atts[0]
                att = att.repeat_interleave(2, -2)
                att_img = att.detach().unsqueeze(0).cpu().numpy()
                att_img -= att_img.min()
                att_img = (att_img / att_img.max() * 255).astype(np.uint8)
                cur_img = (img[0].cpu().numpy() * 255).astype(np.uint8)

                writer.add_image(stage + '-img-' + str(img_count), cur_img, epoch)
                writer.add_image(stage + '-att-' + str(img_count), att_img, epoch)

                img_count += 1

            # print
            if i == 0:
                gt_text = gt_text[0][:lens[0]]
                pd_text = loaders[stage].dataset.tensor2text(preds[0][:lens[0]])

                print('\nGT:', gt_text)
                print('PD:', pd_text)

            # backward
            if is_train:
                loss.backward()
                optim.step()

    return metrics


def main():
    """Application entry point."""
    args = get_args()

    # make all dirs
    models_dir = args.save_to.joinpath('models')
    models_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = args.save_to.joinpath('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(logs_dir)

    # datasets
    c2i, i2c = build_alphabet(light=False, with_ctc_blank=False, with_sos=True)
    text_max_len = 98
    ds_args = {'images_dir': args.images_dir,
               'markup_dir': args.mkp_dir,
               'height': args.height,
               'i2c': i2c,
               'max_len': text_max_len}
    ds_train = IAMDataset(split_filepath=args.train_split,
                          augment=args.augment,
                          **ds_args)
    ds_valid = IAMDataset(split_filepath=args.valid_split, **ds_args)

    dl_args = {'batch_size': args.bs,
               'num_workers': args.workers,
               'shuffle': True}
    loaders = {'train': DataLoader(ds_train,
                                   collate_fn=ds_train.collate_fn, **dl_args),
               'valid': DataLoader(ds_valid,
                                   collate_fn=ds_valid.collate_fn, **dl_args)}
    # model
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = create_model(c2i, i2c, 'seq2seq', ds_train.max_len, args.pos_encoding).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # model saving initialization
    best_metrics = {'cer': {'train': np.inf, 'valid': np.inf},
                    'loss': {'train': np.inf, 'valid': np.inf}}

    # continue training if needed    
    if args.ckpt_path is not None:
        model, optim, best_metrics = load_model(args.ckpt_path, model, optim, device)

    ep = 1
    while ep != args.epochs + 1:
        print(f'\nEpoch #{ep}')
        metrics = epoch_step(model, loaders, device, optim, writer, ep)

        # dump metrics
        for m_name, m_values in metrics.items():
            writer.add_scalars(m_name, m_values, ep)

        # dump best metrics
        for m_name, m_values in best_metrics.items():
            writer.add_scalars('best_' + m_name, m_values, ep)

        # print metrics
        print('Current metrics:')
        pprint(metrics)

        print('Best metrics:')
        pprint(best_metrics)

        # save best models and update best metrics
        save_model(model, optim, args, ep, metrics, best_metrics, models_dir)
        writer.flush()

        ep += 1


if __name__ == '__main__':
    main()
