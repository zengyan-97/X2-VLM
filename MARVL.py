import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100

    for i, (image0, image1, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)

        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        loss = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for image0, image1, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        prediction = model(images, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    train_dataset, val_dataset, test_dataset_dict = create_dataset('marvl', config)
    datasets = [train_dataset, val_dataset]

    train_dataset_size = len(train_dataset)
    train_batch_size = config['batch_size']
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")
        print(f"### Test: {[(k, len(dataset)) for k, dataset in test_dataset_dict.items()]}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']] * 2,
                                             num_workers=[4, 4], is_trains=[True, False],
                                             collate_fns=[None, None])

    test_loader_dict = {}
    for k, v in test_dataset_dict.items():
        test_loader_dict[k] = create_loader([v], [None], batch_size=[config['batch_size']],
                                            num_workers=[4], is_trains=[False], collate_fns=[None])[0]

    print("Creating model")
    from models.model_classification import XVLMPlusForMARVL
    model = XVLMPlusForMARVL(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    tokenizer = build_tokenizer(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")

        acc_mean = 0
        for language, test_loader in test_loader_dict.items():
            test_stats = evaluate(model, test_loader, tokenizer, device)
            if utils.is_main_process():
                print({f'test_{language}_{k}': v for k, v in test_stats.items()}, flush=True)
                acc_mean += (test_stats['acc'] / len(test_loader_dict))

            dist.barrier()

        if utils.is_main_process():
            print("Test average accuracy: ", acc_mean, flush=True)
        dist.barrier()

    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (train_batch_size * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']

        best = 0
        best_epoch = 0
        if 'eval_interval' not in config:
            config['eval_interval'] = 1

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler)
            if epoch >= config['start_eval']:
                # val_stats = evaluate(model, val_loader, tokenizer, device)

                acc_mean = 0
                for language, test_loader in test_loader_dict.items():
                    test_stats = evaluate(model, test_loader, tokenizer, device)
                    if utils.is_main_process():
                        print({f'test_{language}_{k}': v for k, v in test_stats.items()}, flush=True)
                        acc_mean += (float(test_stats['acc']) / len(test_loader_dict))
                    dist.barrier()

                if utils.is_main_process():
                    if acc_mean > best:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            # 'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = acc_mean
                        best_epoch = epoch

                    print("best epoch: {:}, best test acc_mean: {:.4f}".format(best_epoch, best), flush=True)

            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/MARVL.yaml')
    parser.add_argument('--output_dir', default='output/nlvr')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--load_nlvr_pretrain', action='store_true')
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--lr', default=0., type=float)
    parser.add_argument('--fewshot', default='', type=str)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.lr != 0.:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr
    if args.fewshot:
        config['train_file'][0] = config['train_file'][0].format(args.fewshot)
        config['val_file'][0] = config['val_file'][0].format(args.fewshot)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)