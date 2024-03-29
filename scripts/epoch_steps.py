#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Training epoch for different models.."""
import torch
import numpy as np
from tqdm import tqdm
from common.utils import calc_cer


def vis_attn(atts, img, writer, img_count, epoch, stage):
    if len(atts[0].shape) == 2:
        att = atts[0].detach().repeat_interleave(2, -2)
    else:  # == 3
        att = atts[0].detach().sum(dim=1).repeat_interleave(2, -2)

    att_img = att.unsqueeze(0).cpu().numpy()
    att_img -= att_img.min()
    att_img = (att_img / att_img.max() * 255).astype(np.uint8)
    cur_img = (img[0].cpu().numpy() * 255).astype(np.uint8)

    writer.add_image(stage + '-img-' + str(img_count), cur_img, epoch)
    writer.add_image(stage + '-att-' + str(img_count), att_img, epoch)


def get_epoch_step_fn(model_type='baseline'):
    if model_type in ['seq2seq', 'seq2seq_light']:
        return epoch_step_seq2seq
    else:
        return epoch_step_baseline


def get_metrics_dict(model_type='baseline', init_value=0.0):
    if model_type in ['seq2seq', 'seq2seq_light']:
        return {'cer': {'train': init_value, 'valid': init_value},
                'loss': {'train': init_value, 'valid': init_value}}
    else:
        return {'cer': {'train': init_value, 'valid': init_value},
                'cer-beam': {'train': init_value, 'valid': init_value},
                'loss': {'train': init_value, 'valid': init_value}}


def epoch_step_baseline(model, loaders, device, optim, *args, **kwargs):
    metrics = get_metrics_dict(model_type='baseline')

    for stage in ['train', 'valid']:
        is_train = stage == 'train'
        model.train() if is_train else model.eval()
        loader_size = len(loaders[stage])

        torch.set_grad_enabled(is_train)
        for i, (img, txt, lens) in enumerate(tqdm(loaders[stage], desc=stage)):
            if is_train:
                optim.zero_grad()

            # forward
            img, txt, lens = img.to(device), txt.to(device), lens.to(device)
            logits, log_probs = model(img)

            # loss
            loss = model.calc_loss(logits, log_probs, txt, lens)
            metrics['loss'][stage] += loss.item() / loader_size

            # cer
            gt_text = loaders[stage].dataset.tensor2text(txt)
            pd_beam, _ = model.decode(log_probs)
            pd_beam = loaders[stage].dataset.tensor2text(pd_beam)

            gt_lens = lens.detach().cpu().numpy()
            cer = calc_cer(gt_text, pd_beam, gt_lens)
            metrics['cer-beam'][stage] += cer / loader_size

            preds = torch.argmax(logits, dim=1).permute(1, 0)
            pd_text = loaders[stage].dataset.tensor2text(preds.permute(1, 0))
            cer = calc_cer(gt_text, pd_text, gt_lens)
            metrics['cer'][stage] += cer / loader_size

            # print
            if i == 0:
                gt_text = gt_text[0][:lens[0]]
                pd_text = loaders[stage].dataset.tensor2text(
                    preds[:, 0][:lens[0]]
                )
                pd_beam = pd_beam[0][:lens[0]]

                print('\nGT:', gt_text)
                print('PD:', pd_text)
                print('PD beam:', pd_beam)

            # backward
            if is_train:
                loss.backward()
                optim.step()

    return metrics


def epoch_step_seq2seq(model, loaders, device, optim, writer, epoch):
    metrics = get_metrics_dict(model_type='seq2seq')

    for stage in ['train', 'valid']:
        is_train = stage == 'train'
        model.train() if is_train else model.eval()
        loader_size = len(loaders[stage])

        torch.set_grad_enabled(is_train)
        img_count = 0
        for i, (img, txt, lens) in enumerate(tqdm(loaders[stage], desc=stage)):
            if is_train:
                optim.zero_grad()

            # forward
            img, txt, lens = img.to(device), txt.to(device), lens.to(device)
            logs_probs, preds, atts = model(img, txt)

            # loss
            loss = model.calc_loss(logs_probs, txt, lens)
            metrics['loss'][stage] += loss.item() / loader_size

            # cer
            gt_text = loaders[stage].dataset.tensor2text(txt)
            gt_lens = lens.detach().cpu().numpy()

            pd_text = loaders[stage].dataset.tensor2text(preds)
            cer = calc_cer(gt_text, pd_text, gt_lens)
            metrics['cer'][stage] += cer / loader_size

            # dump attention
            if i % 50 == 0:
                vis_attn(atts, img, writer, img_count, epoch, stage)
                img_count += 1

            # print
            if i == 0:
                gt_text = gt_text[0][:lens[0]]
                pd_text = loaders[stage].dataset.tensor2text(
                    preds[0][:lens[0]]
                )

                print('\nGT:', gt_text)
                print('PD:', pd_text)

            # backward
            if is_train:
                loss.backward()
                optim.step()

    return metrics
