#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Training launcher."""
import sys
import torch
import configargparse

import numpy as np

from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from iam_dataset import IAMDataset
from epoch_steps import get_epoch_step_fn, get_metrics_dict

sys.path.append(str(Path(sys.path[0]).parent))
from common.utils import build_alphabet, create_model, print_model_params
from common.conv_net import get_models_list


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', is_config_file=True,
                        help='Path to config file.')

    # dataset
    parser.add_argument('--model-type', required=True,
                        choices=['baseline', 'seq2seq',
                                 'seq2seq_light', 'seg_attn'],
                        help='Model type to train.')
    parser.add_argument('--ckpt-path', type=Path,
                        help='Path to saved model to load for training.')
    parser.add_argument('--load-model-only', action='store_true',
                        help='Load only model from checkpoint.')
    parser.add_argument('--load-fe-only', action='store_true',
                        help='Load only Feature Extractor from checkpoint.')
    parser.add_argument('--freeze-backbone', type=int, default=-1,
                        help='Freeze pretrained backbone for this number of '
                             'epochs. Pass -1 for infinite freeze.'
                             'Pass 0 to disable freezing.')
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
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers.')
    parser.add_argument('--augment', action='store_true',
                        help='Augment images.')
    parser.add_argument('--correction', action='store_true',
                        help='Correct slant, skew and contrast.')
    parser.add_argument('--text-max-len', type=int, default=98,
                        help='Max length of text.')
    parser.add_argument('--height', type=int, default=64,
                        help='Input image height. Will resize to this value.')
    parser.add_argument('--img-max-width', type=int, default=1408,
                        help='Max width of images. '
                             'Needed for stable validation process.')

    # common models args
    parser.add_argument('--backbone-out', type=int, default=256,
                        help='Backbone out channels number (for custom only).')
    parser.add_argument('--fe-dropout', type=float, default=0.15,
                        help='Dropout for the end of Backbone.')
    parser.add_argument('--dec-dropout', type=float, default=0.15,
                        help='Dropout for the Decoder.')
    parser.add_argument('--dec-hs', type=int, default=256,
                        help='Decoder hidden size.')
    parser.add_argument('--dec-layers', type=int, default=1,
                        help='Decoder RNN layers.')
    parser.add_argument('--rnn-dropout', type=float, default=0.0,
                        help='Dropout inside rnns.')

    parser.add_argument('--save-to', type=Path,
                        help='Path to save dir.')

    # add baseline args
    subparsers = parser.add_subparsers()
    baseline = subparsers.add_parser('baseline-args')
    baseline.add_argument('--config-baseline', is_config_file=True,
                          help='Path to baseline config file.')
    baseline.add_argument('--n-layers', type=int, default=2,
                          help='Number of RNN layers.')

    # add seq2seq args
    seq2seq = subparsers.add_parser('seq2seq-args')
    seq2seq.add_argument('--config-seq2seq', is_config_file=True,
                        help='Path to seq2seq config file.')
    seq2seq.add_argument('--pos-encoding', action='store_true',
                         help='Add positional encoding before decoder.')
    seq2seq.add_argument('--enc-hs', type=int, default=128,
                         help='(bidirectional) Encoder hidden size.')
    seq2seq.add_argument('--attn-size', type=int, default=256,
                         help='Attention size.')
    seq2seq.add_argument('--emb-size', type=int, default=128,
                         help='Embedding size.')
    seq2seq.add_argument('--teacher-rate', type=float, default=1.0,
                         help='Teacher rate for decoder training input.')
    seq2seq.add_argument('--enc-layers', type=int, default=1,
                         help='Encoder RNN layers.')
    seq2seq.add_argument('--backbone', default='conv_net6',
                         choices=get_models_list(),
                         help='Backbone type. Use carefully.')

    # add seg_attn args
    seg_attn = subparsers.add_parser('seg-attn-args')
    seg_attn.add_argument('--config-seg-attn', is_config_file=True,
                          help='Path to seg_attn config file.')
    seg_attn.add_argument('--teacher-rate', type=float, default=1.0,
                          help='Teacher rate for decoder training input.')
    seg_attn.add_argument('--emb-size', type=int, default=256,
                          help='Embedding size.')
    seg_attn.add_argument('--backbone', default='custom',
                          choices=['custom', 'resnet18',
                                   'resnet34', 'efficientnet_b0'],
                          help='Backbone type.')
    seg_attn.add_argument('--pos-encoding', action='store_true',
                          help='Add positional encoding before decoder.')

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


def load_ckpt(ckpt_path,
              model,
              optim,
              device,
              model_type,
              model_only=False,
              fe_only=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    ep = 1

    if model_only:
        print('\nLoading model only')
        model.load_state_dict(ckpt['model'])
        best_metrics = get_metrics_dict(model_type, init_value=np.inf)
    elif fe_only:
        print('\nLoading Features Extractor only')
        sd = {k.split('.', maxsplit=1)[1]: v
                for k, v in ckpt['model'].items()
                if k.split('.')[0] == 'fe'}
        model.fe.load_state_dict(sd)
        best_metrics = get_metrics_dict(model_type, init_value=np.inf)
    else:
        print('\nLoading model, optim and best metrics')
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        best_metrics = ckpt['best_metrics']
        ep = ckpt['epoch'] + 1

    print('\nCheckpoint metrics:')
    pprint(ckpt['metrics'])

    return model, optim, best_metrics, ep


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

    if args.model_type in ['seq2seq', 'seg_attn', 'seq2seq_light']:
        c2i, i2c = build_alphabet(light=False,
                                  with_ctc_blank=False,
                                  with_sos=True)
    else:
        c2i, i2c = build_alphabet(light=False,
                                  with_ctc_blank=True,
                                  with_sos=False)

    epoch_step = get_epoch_step_fn(args.model_type)
    model = create_model(c2i, i2c, args=args).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # datasets
    ds_args = {'images_dir': args.images_dir,
               'markup_dir': args.mkp_dir,
               'height': args.height,
               'i2c': i2c,
               'max_len': args.text_max_len,
               'correction': args.correction}
    ds_train = IAMDataset(split_filepath=args.train_split,
                          augment=args.augment,
                          **ds_args)
    ds_valid = IAMDataset(split_filepath=args.valid_split,
                          width=args.img_max_width,
                          **ds_args)

    dl_args = {'batch_size': args.bs,
               'num_workers': args.workers}
    loaders = {'train': DataLoader(ds_train, shuffle=True,
                                   collate_fn=ds_train.collate_fn, **dl_args),
               'valid': DataLoader(ds_valid,
                                   collate_fn=ds_valid.collate_fn, **dl_args)}

    # model saving initialization
    best_metrics = get_metrics_dict(model_type=args.model_type,
                                    init_value=np.inf)

    # continue training if needed
    ep = 1
    if args.ckpt_path is not None:
        model, optim, best_metrics, ep = load_ckpt(
            args.ckpt_path,
            model,
            optim,
            device,
            args.model_type,
            args.load_model_only,
            args.load_fe_only
        )

        # freeze pretrained backbone
        if args.freeze_backbone != 0:
            model.fe.freeze()

    # print the number of model's parameters
    print_model_params(model)

    cur_training_epochs = 0  # number of completed epochs for the current launch
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
        cur_training_epochs += 1

        # defrost if needed
        if cur_training_epochs == args.freeze_backbone:
            model.fe.defrost()


if __name__ == '__main__':
    main()
