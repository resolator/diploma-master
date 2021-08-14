#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Training launcher."""
import torch
import fastwer
import configargparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from iam_dataset import IAMDataset
from baseline_net_pytorch import BaselineNet
from baseline_net_my import BaselineNet as MyBaselineNet


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

    parser.add_argument('--epochs', type=int, default=-1,
                        help='Number of epochs to train. '
                             'Pass -1 for infinite training.')
    parser.add_argument('--bs', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of data loader workers.')
    parser.add_argument('--height', type=int, default=64,
                        help='Input image height. Will resize to this value.')
    parser.add_argument('--augment', action='store_true',
                        help='Augment images.')

    parser.add_argument('--model-type', default='my',
                        choices=['my', 'pytorch'],
                        help='Model implementation to train.')
    parser.add_argument('--teacher-ratio', type=float, default=0.9,
                        help='Teacher ratio for decoder.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout.')

    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    return parser.parse_args()


def create_model(height=64,
                 enc_hs=256,
                 dec_hs=256,
                 enc_n_layers=3,
                 enc_bidirectional=True,
                 max_len=150,
                 teacher_ratio=0.9,
                 model_type='my',
                 dropout=0.5):
    """Wrapper for creating different models."""
    if model_type == 'my':
        model_class = MyBaselineNet
    else:
        model_class = BaselineNet

    model = model_class(height=height,
                        enc_hs=enc_hs,
                        dec_hs=dec_hs,
                        enc_n_layers=enc_n_layers,
                        enc_bidirectional=enc_bidirectional,
                        max_len=max_len,
                        teacher_ratio=teacher_ratio,
                        dropout=dropout)

    return model


def save_model(model, optim, args, ep, metrics, best_metrics, models_dir):
    """Save best models and update best metrics."""
    save_data = {'model': model.state_dict(),
                 'optim': optim.state_dict(),
                 'args': args,
                 'epoch': ep,
                 'valid_metrics': metrics,
                 'i2c': model.i2c}

    for m in metrics.keys():
        for stage in ['train', 'valid']:
            if metrics[m][stage] < best_metrics[m][stage]:
                model_path = models_dir.joinpath(stage + '-' + m + '.pth')
                torch.save(save_data, model_path)
                best_metrics[m][stage] = metrics[m][stage]

                print(f'Saved {stage} {m}')


def calc_cer(gt, pd, gt_lens=None):
    if gt_lens is not None:
        gt = [x[:y] for x, y in zip(gt, gt_lens)]
        pd = [x[:y] for x, y in zip(pd, gt_lens)]

    return fastwer.score(pd, gt, char_level=True)


def epoch_step(model, loaders, device, optim):
    metrics = {'cer': {'train': 0.0, 'valid': 0.0},
               'cer_beam': {'train': 0.0, 'valid': 0.0},
               'loss': {'train': 0.0, 'valid': 0.0}}
    my_baseline_net = False

    for stage in ['train', 'valid']:
        is_train = stage == 'train'
        model.train() if is_train else model.eval()
        loader_size = len(loaders[stage])

        torch.set_grad_enabled(is_train)
        for i, (img, text, lens) in enumerate(tqdm(loaders[stage], desc=stage)):
            if is_train:
                optim.zero_grad()

            # old
            img, text, lens = img.to(device), text.to(device), lens.to(device)

            if my_baseline_net:
                logits, preds = model(img, text, lens)
                loss = model.calc_loss(logits, text, lens, preds, ce=True)
            else:
                logits, loss = model(img, text, lens)
                preds = torch.argmax(logits.permute(1, 0, 2),
                                     dim=-1).permute(1, 0)

            # loss
            metrics['loss'][stage] += loss.item() / loader_size

            # cer
            gt_text = loaders[stage].dataset.tensor2text(text)
            pd_lens = model.calc_preds_lens(preds)
            pd_beam = model.decode_ctc_beam_search(logits, pd_lens)
            pd_beam = loaders[stage].dataset.tensor2text(pd_beam, True)

            gt_lens = lens.detach().cpu().numpy()
            cer = calc_cer(gt_text, pd_beam, gt_lens)
            metrics['cer_beam'][stage] += cer / loader_size

            pd_text = loaders[stage].dataset.tensor2text(preds.permute(1, 0))
            cer = calc_cer(gt_text, pd_text, gt_lens)
            metrics['cer'][stage] += cer / loader_size

            # print
            if i == 0:
                gt_text = gt_text[0][:lens[0]]
                pd_text = loaders[stage].dataset.tensor2text(preds[:, 0][:lens[0]])
                pd_beam = pd_beam[0][:lens[0]]

                print('\nGT:', gt_text)
                print('PD:', pd_text)
                print('PD beam:', pd_beam)

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

    # model
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    text_max_len = 150
    model = create_model(
        height=args.height,
        enc_hs=256,
        dec_hs=256,
        max_len=text_max_len,
        teacher_ratio=args.teacher_ratio,
        model_type=args.model_type
    ).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # datasets
    ds_args = {'images_dir': args.images_dir,
               'markup_dir': args.mkp_dir,
               'height': args.height,
               'c2i': model.c2i,
               'max_len': text_max_len}
    ds_train = IAMDataset(split_filepath=args.train_split,
                          augment=args.augment,
                          **ds_args)
    ds_valid = IAMDataset(split_filepath=args.valid_split, **ds_args)

    dl_args = {'batch_size': args.bs,
               'num_workers': args.workers,
               'collate_fn': IAMDataset.collate_fn}
    loaders = {
        'train': DataLoader(ds_train, shuffle=True, **dl_args),
        'valid': DataLoader(ds_valid, **dl_args)
    }

    # model saving initialization
    best_metrics = {'cer': {'train': np.inf, 'valid': np.inf},
                    'cer_beam': {'train': np.inf, 'valid': np.inf},
                    'loss': {'train': np.inf, 'valid': np.inf}}

    ep = 1
    while ep != args.epochs + 1:
        print(f'\nEpoch #{ep}')
        metrics = epoch_step(model, loaders, device, optim)

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
