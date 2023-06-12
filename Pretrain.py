# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.
import argparse
import os
import sys

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer

import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy
from accelerators.apex_ddp_accelerator import ApexDDPAccelerator


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        # from transformers import get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)


def run_image_iter(model, image_batch, optimizer, accelerator, metric_logger, device,
                   ret_match_loss=True, return_loss_only=False):
    image, batch = image_batch[0].to(device, non_blocking=True), [t.to(device) if t is not None else None for t in image_batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = batch

    loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked,
                 masked_pos=masked_pos, masked_ids=masked_ids, ret_match_loss=ret_match_loss)

    if return_loss_only:
        return loss

    optimizer.zero_grad()
    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_itc=loss['loss_itc'].item())
    metric_logger.update(loss_itm=loss['loss_itm'].item())
    metric_logger.update(loss_mlm=loss['loss_mlm'].item())


def run_region_iter(model, region_batch, optimizer, accelerator, metric_logger, device,
                    ret_match_loss=True, return_loss_only=False):
    image, region_batch = region_batch[0].to(device, non_blocking=True), [
        t.to(device) if t is not None else None for t in region_batch[1:]]

    idx_to_group_img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
    image_atts, target_bbox, is_image = region_batch

    if config['calc_image_bbox_loss']:
        is_image = None

    loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos,
                 masked_ids=masked_ids,
                 image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox, is_image=is_image,
                 ret_bbox_loss=True, ret_match_loss=ret_match_loss)

    if return_loss_only:
        return loss

    optimizer.zero_grad()
    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm'] + loss['loss_bbox'] + loss['loss_giou']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_ritc=loss['loss_itc'].item())
    metric_logger.update(loss_ritm=loss['loss_itm'].item())
    metric_logger.update(loss_rmlm=loss['loss_mlm'].item())
    metric_logger.update(loss_rbbox=loss['loss_bbox'].item())
    metric_logger.update(loss_rgiou=loss['loss_giou'].item())


def run_video_iter(model, video_batch, optimizer, accelerator, metric_logger, device,
                   ret_match_loss=True, return_loss_only=False):
    frames, batch = video_batch[0].to(device, non_blocking=True), [t.to(device) if t is not None else None for t in video_batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = batch

    loss = model(frames, text_ids, text_atts, text_ids_masked=text_ids_masked,
                 masked_pos=masked_pos, masked_ids=masked_ids, ret_match_loss=ret_match_loss)

    if return_loss_only:
        return loss

    optimizer.zero_grad()
    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_vitc=loss['loss_itc'].item())
    metric_logger.update(loss_vitm=loss['loss_itm'].item())
    metric_logger.update(loss_vmlm=loss['loss_mlm'].item())


def run_text_iter(model, batch, optimizer, accelerator, metric_logger, device, return_loss_only=False):
    batch = [t.to(device) if t is not None else None for t in batch]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = batch

    loss = model(None, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids)

    if return_loss_only:
        return loss

    optimizer.zero_grad()
    loss_in_total = loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_tmlm=loss['loss_mlm'].item())


def run_mtext_iter(model, batch, optimizer, accelerator, metric_logger, device, return_loss_only=False):
    batch = [t.to(device) if t is not None else None for t in batch]

    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
    text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = batch

    loss = model(None, text_ids=text_ids, text_atts=text_atts, text_ids_masked=text_ids_masked,
                 masked_pos=masked_pos, masked_ids=masked_ids, text_ids_2=text_ids_2,
                 text_atts_2=text_atts_2, text_ids_masked_2=text_ids_masked_2,
                 masked_pos_2=masked_pos_2, masked_ids_2=masked_ids_2)

    if return_loss_only:
        return loss

    optimizer.zero_grad()

    loss_in_total = loss['loss_ttc'] + loss['loss_ttm'] + loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_tt_ttc=loss['loss_ttc'].item())
    metric_logger.update(loss_tt_ttm=loss['loss_ttm'].item())
    metric_logger.update(loss_tt_mlm=loss['loss_mlm'].item())


def run_mixed_iter(model, image_batch, region_batch, text_batch, video_batch, mtext_batch,
                   optimizer, accelerator, metric_logger, device, ret_match_loss=True):
    optimizer.zero_grad()

    if video_batch is not None:
        v_loss = run_video_iter(model, video_batch, optimizer, accelerator, metric_logger, device,
                   ret_match_loss=ret_match_loss, return_loss_only=True)

        accelerator.backward_step(config['videos'].get('iter_perc', 1.0) * (
                v_loss['loss_itc'] + v_loss['loss_itm'] + v_loss['loss_mlm']), optimizer)

        metric_logger.update(loss_vitc=v_loss['loss_itc'].item())
        metric_logger.update(loss_vitm=v_loss['loss_itm'].item())
        metric_logger.update(loss_vmlm=v_loss['loss_mlm'].item())

    i_loss = run_image_iter(model, image_batch, optimizer, accelerator, metric_logger, device,
                   ret_match_loss=ret_match_loss, return_loss_only=True)
    loss_in_total = config['images'].get('iter_perc', 1.0) * (i_loss['loss_itc'] + i_loss['loss_itm'] + i_loss['loss_mlm'])

    metric_logger.update(loss_itc=i_loss['loss_itc'].item())
    metric_logger.update(loss_itm=i_loss['loss_itm'].item())
    metric_logger.update(loss_mlm=i_loss['loss_mlm'].item())

    if region_batch is not None:
        r_loss = run_region_iter(model, region_batch, optimizer, accelerator, metric_logger, device,
                        ret_match_loss=ret_match_loss, return_loss_only=True)

        if config.get('regions_use_bbox_only', False):
            loss_in_total = loss_in_total + config['regions'].get('iter_perc', 1.0) * (
                    r_loss['loss_bbox'] + r_loss['loss_giou'])
        else:
            loss_in_total = loss_in_total + config['regions'].get('iter_perc', 1.0) * (
                        r_loss['loss_itc'] + r_loss['loss_itm'] +
                        r_loss['loss_mlm'] + r_loss['loss_bbox'] +
                        r_loss['loss_giou'])

            metric_logger.update(loss_ritc=r_loss['loss_itc'].item())
            metric_logger.update(loss_ritm=r_loss['loss_itm'].item())
            metric_logger.update(loss_rmlm=r_loss['loss_mlm'].item())

        metric_logger.update(loss_rbbox=r_loss['loss_bbox'].item())
        metric_logger.update(loss_rgiou=r_loss['loss_giou'].item())

    if text_batch is not None:
        t_loss = run_text_iter(model, text_batch, optimizer, accelerator, metric_logger, device, return_loss_only=True)
        loss_in_total = loss_in_total + config['texts'].get('iter_perc', 1.0) * t_loss['loss_mlm']
        metric_logger.update(loss_tmlm=t_loss['loss_mlm'].item())

    if mtext_batch is not None:
        tt_loss = run_mtext_iter(model, mtext_batch, optimizer, accelerator, metric_logger, device, return_loss_only=True)

        loss_in_total = loss_in_total + config['mtexts'].get('iter_perc', 1.0) * (
                tt_loss['loss_ttc'] + tt_loss['loss_ttm'] + tt_loss['loss_mlm'])

        metric_logger.update(loss_ttc=tt_loss['loss_ttc'].item())
        metric_logger.update(loss_ttm=tt_loss['loss_ttm'].item())
        metric_logger.update(loss_ttmlm=tt_loss['loss_mlm'].item())

    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()


def train(model, image_loader, region_loader, text_loader, image_loader_aux, video_loader, video_loader_aux, mtext_loader, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer):
    model.train()
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_large', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train step: [{}]'.format(start_epoch)
    assert start_epoch == 0
    print_freq = 50

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size']/(config['batch_size']*world_size))
    assert step_per_epoch > 1
    global_step = 0  # start from 0

    if image_loader_aux is not None:
        image_iter_aux = iter(image_loader_aux)  # cleaner data
    else:
        image_iter_aux = None

    if video_loader is not None:
        video_iter = iter(video_loader)
        metric_logger.add_meter('loss_vitc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_vitm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_vmlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    else:
        video_iter = None

    if video_loader_aux is not None:
        video_iter_aux = iter(video_loader_aux)  # cleaner data
    else:
        video_iter_aux = None

    if region_loader is not None:
        region_iter = iter(region_loader)
        if not config.get('regions_use_bbox_only', False):
            metric_logger.add_meter('loss_ritc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
            metric_logger.add_meter('loss_ritm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
            metric_logger.add_meter('loss_rmlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

        metric_logger.add_meter('loss_rbbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_rgiou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        region_iter = None

    if text_loader is not None:
        text_iter = iter(text_loader)
        metric_logger.add_meter('loss_tmlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        text_iter = None

    if mtext_loader is not None:
        # parallel texts
        mtext_iter = iter(mtext_loader)
        metric_logger.add_meter('loss_ttc', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_ttm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_ttmlm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))

    else:
        mtext_iter = None

    stop_calc_itm = config.get('stop_calc_itm', float('inf'))  # steps
    print(f"### Stop Calculate Matching Loss After {stop_calc_itm} Steps", flush=True)

    for i, batch in enumerate(metric_logger.log_every(image_loader, print_freq, header, step_per_epoch, epoch_info)):

        with torch.no_grad():
            model.module.temp.clamp_(0.001, 0.5)

        if config.get('mixed_in_batch', False):

            if image_iter_aux is not None:  # if having cleaner data
                ret_match_loss = False  # do not calc matching loss on noisy data
                if random.random() < config['aux_iter_perc']:
                    batch = next(image_iter_aux)  # 这个实现可能不那么好, 在1个epoch中, 会丢一些 noisy data
                    ret_match_loss = global_step < stop_calc_itm

            else:
                ret_match_loss = global_step < stop_calc_itm

            if video_iter_aux is not None:
                assert video_iter is not None
                if random.random() < config['video_aux_iter_perc']:
                    video_batch = next(video_iter_aux)
                else:
                    video_batch = next(video_iter)

            else:
                video_batch = next(video_iter) if video_iter is not None else None

            region_batch = next(region_iter) if region_iter is not None else None

            text_batch = next(text_iter) if text_iter is not None else None
            mtext_batch = next(mtext_iter) if mtext_iter is not None else None

            run_mixed_iter(model, batch, region_batch, text_batch, video_batch, mtext_batch,
                           optimizer, accelerator, metric_logger, device, ret_match_loss=ret_match_loss)

        else:
            raise ValueError("i didn't use this")

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_large=optimizer.param_groups[2]["lr"])
        scheduler.step()

        current_epoch = global_step // step_per_epoch
        if (global_step+1) % step_per_epoch == 0:
            if utils.is_main_process():
                train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch,
                             }

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if ((current_epoch+1) % config['ckpt_frequent'] == 0) or (current_epoch+1 == config['schedular']['epochs']):
                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': scheduler.state_dict(),
                        'config': config,
                        # 'epoch': current_epoch,
                    }
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch=current_epoch,
                                                 training_states=optimizer.state_dict())

            dist.barrier()

        if (global_step+1) % config['ckpt_frequent_step'] == 0:
            if utils.is_main_process():
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': scheduler.state_dict(),
                    'config': config,
                    # 'epoch': current_epoch,
                }

                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=current_epoch, step=global_step,
                                             training_states=optimizer.state_dict())

            dist.barrier()

        if config['schedular'].get('num_training_steps', False) and (global_step+1 >= config['schedular']['num_training_steps']):
            break

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    config['batch_size'] = config['images']['batch_size']

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset", flush=True)
    image_dataset, region_dataset, text_dataset, image_dataset_aux, video_dataset, video_dataset_aux, mtext_dataset = create_dataset('pretrain', config)

    if utils.is_main_process():
        print(f"### images: {config['train_file']}", flush=True)
        print(f"### images_aux: {config.get('train_file_aux', '')}", flush=True)
        print(f"### regions: {config.get('train_file_regions', '')}", flush=True)
        print(f"### texts: {config.get('train_file_text', '')}", flush=True)
        print(f"### videos: {config.get('train_file_videos', '')}", flush=True)
        print(f"### videos_aux: {config.get('train_file_videos_aux', '')}", flush=True)
        print(f"### mtexts: {config.get('train_file_mtext', '')}", flush=True)
        print(f"### batch size, {config['batch_size']} x {int(os.environ.get('WORLD_SIZE', 1))}")

    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=config['images']['batch_size'],
                                               num_workers=config['images']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=image_dataset.collate_fn)

    if video_dataset is not None:
        video_loader = torch.utils.data.DataLoader(video_dataset,
                                                   batch_size=config['videos']['batch_size'],
                                                   num_workers=config['videos']['num_workers'],
                                                   pin_memory=True,
                                                   drop_last=False,
                                                   collate_fn=video_dataset.collate_fn)
    else:
        video_loader = None

    if video_dataset_aux is not None:
        video_loader_aux = torch.utils.data.DataLoader(video_dataset_aux,
                                                   batch_size=config['videos']['batch_size'],
                                                   num_workers=config['videos']['num_workers'],
                                                   pin_memory=True,
                                                   drop_last=False,
                                                   collate_fn=video_dataset_aux.collate_fn)
    else:
        video_loader_aux = None

    if image_dataset_aux is not None:  # for small-scale high-quality images
        image_loader_aux = torch.utils.data.DataLoader(image_dataset_aux,
                                                       batch_size=config['images']['batch_size'],
                                                       num_workers=config['images']['num_workers'],
                                                       pin_memory=True,
                                                       drop_last=False,
                                                       collate_fn=image_dataset_aux.collate_fn)
    else:
        image_loader_aux = None

    if region_dataset is not None:
        region_loader = torch.utils.data.DataLoader(region_dataset, batch_size=config['regions']['max_images'],
                                                    # batch_size = max_images * max_regions
                                                    num_workers=config['regions']['num_workers'],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    collate_fn=region_dataset.collate_fn)
    else:
        region_loader = None

    if text_dataset is not None:
        text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=config['texts']['batch_size'],
                                                  num_workers=config['texts']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  collate_fn=text_dataset.collate_fn)
    else:
        text_loader = None

    if mtext_dataset is not None:
        mtext_loader = torch.utils.data.DataLoader(mtext_dataset, batch_size=config['mtexts']['batch_size'],
                                                  num_workers=config['mtexts']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  collate_fn=mtext_dataset.collate_fn)
    else:
        mtext_loader = None

    print(f"Creating model {config.get('model_type', 'XVLM')}", flush=True)
    if config.get('model_type', ''):
        if config['model_type'] == 'XVLMPlus':
            from models.model_pretrain import XVLMPlus

            assert os.path.exists(args.checkpoint)
            model = XVLMPlus(config=config, load_text_params=False, load_vision_params=False, load_cross_params=False, pretraining=False)

            text_ckpt_rpath = ''
            if config.get('replace_text_encoder', False):
                text_ckpt_rpath = os.path.join(config['text_encoder'], 'pytorch_model.bin')

            model.load_pretrained(args.checkpoint, config, text_ckpt_rpath=text_ckpt_rpath)

        elif config['model_type'] == 'CrossViewLM':
            from models.model_pretrain import CrossViewLM

            assert os.path.exists(args.checkpoint)
            model = CrossViewLM(config=config, load_text_params=False, load_vision_params=False, load_cross_params=False,
                             pretraining=False)

            text_ckpt_rpath = ''
            if config.get('replace_text_encoder', False):
                text_ckpt_rpath = os.path.join(config['text_encoder'], 'pytorch_model.bin')

            model.load_pretrained(args.checkpoint, config, text_ckpt_rpath=text_ckpt_rpath)

        else:
            raise ValueError(f"config['model_type'] == {config['model_type']}")

    else:

        from models.model_pretrain import XVLM

        if os.path.exists(args.checkpoint):  # for domain pre-training
            model = XVLM(config=config, load_text_params=False, load_vision_params=False, pretraining=False)
            model.load_pretrained(args.checkpoint, config, is_domain_pretrain=True)
        else:
            model = XVLM(config=config)

    # print(model)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    arg_sche['min_rate'] = config['min_lr'] / arg_opt['lr'] if 'min_lr' in config else 0
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = ApexDDPAccelerator(arg_acc, logger=None)

    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)
    reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)

    checkpointer = Checkpointer(args.output_dir)

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    epoch_info = (start_epoch, max_epoch)

    if config.get('replace_text_encoder', False):
        if utils.is_main_process():
            print("### Replaced Text Encoder & Saving Zero-Shot Ckpt")
            model_without_ddp = model
            if hasattr(model, 'module'):
                model_without_ddp = model.module

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'config': config,
            }
            checkpointer.save_checkpoint(model_state=save_obj,
                                         epoch='zeroshot',
                                         training_states=optimizer.state_dict())

        dist.barrier()

    print("Start training", flush=True)
    train(model, image_loader, region_loader, text_loader, image_loader_aux, video_loader, video_loader_aux, mtext_loader, optimizer, epoch_info, device, lr_scheduler, config,
          accelerator, checkpointer)
    dist.barrier()

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)

    print('### Time {}'.format(total_time_str))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    utils.update_config(config, args.override_cfg)
    if utils.is_main_process():
        print('config:', json.dumps(config))

    hmkdir(args.output_dir)

    yaml.dump(config, open('config.yaml', 'w'))
    hcopy('config.yaml', args.output_dir)

    main(args, config)