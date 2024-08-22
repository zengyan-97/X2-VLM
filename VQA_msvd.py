"""
MSVD-QA
1500-way classification task.
test set with ground-truth label.
"""
import argparse
import os
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_classification import XVLMForClassification

import utils
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hexists, hcopy

from dataset.utils import collect_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn, build_tokenizer

from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    accumulate_steps = int(config.get('accumulate_steps', 1))
    for i, (image, question, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        targets = targets.to(device)

        loss = model(image, question_input.input_ids, question_input.attention_mask,
                     targets=targets, train=True)
        
        if accumulate_steps > 1:
            loss = loss / accumulate_steps
        
        # backward
        loss.backward()

        if (i+1) % accumulate_steps == 0:
            # update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    
    for image, question, targets in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
        targets = targets.to(device)

        predictions = model(image, question_input.input_ids, question_input.attention_mask, train=False)

        _, pred_class = predictions.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)
        metric_logger.meters['acc'].update(accuracy.item(), n=image.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0

    print("Creating MSVD-QA datasets")
    train_dataset, valid_dataset, test_dataset = create_dataset('vqa_msvd', config, args.evaluate)

    tokenizer = build_tokenizer(config['text_encoder'])
    model = XVLMForClassification(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:
        print("Start evaluating")
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([test_dataset], [False], num_tasks, global_rank)
        else:
            samplers = [None]

        test_loader = create_loader([test_dataset], samplers,
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4], is_trains=[False],
                                    collate_fns=[None])[0]

        test_stats = evaluation(model, test_loader, tokenizer, device)
        if utils.is_main_process():
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
            print(log_stats)

        dist.barrier()

    else:
        print("Start training")
        datasets = [train_dataset, valid_dataset, test_dataset]
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
        else:
            samplers = [None, None, None]

        train_dataset_size = len(train_dataset)
        world_size = utils.get_world_size()

        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

        train_loader, valid_loader, test_loader = create_loader(datasets, samplers,
                                                  batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                                  num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                  collate_fns=[None, None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        accumulate_steps = int(config.get('accumulate_steps', 1))
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size) / accumulate_steps)
        arg_sche['min_rate'] = config['min_lr'] / arg_opt['lr'] if 'min_lr' in config else 0
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        checkpointer = Checkpointer(args.output_dir)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        max_epoch = config['schedular']['epochs']

        best_epoch = 0
        best = 0

        for epoch in range(start_epoch, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            if epoch >= config['start_eval']:
                # val_stats = evaluation(model, valid_loader, tokenizer, device)
                test_stats = evaluation(model, test_loader, tokenizer, device)
            else:
                # val_stats = {}
                test_stats = {}

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             # **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                print(log_stats, flush=True)
                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if (epoch >= config['start_eval']) and (float(test_stats['acc']) > best):
                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch='best',
                                                 training_states=optimizer.state_dict())

                    best = float(test_stats['acc'])
                    best_epoch = epoch
                    print("### Best Epoch: ", best_epoch, flush=True)

            dist.barrier()

        if utils.is_main_process():
            with open("log.txt", "a") as f:
                f.write("best epoch: %d" % best_epoch)

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/vqa2_base.yaml')
    parser.add_argument('--output_dir', default='output/vqa')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    utils.update_config(config, args.override_cfg)
    if utils.is_main_process():
        print('config:', json.dumps(config))

    args.result_dir = os.path.join(args.output_dir, 'result')
    hmkdir(args.output_dir)
    hmkdir(args.result_dir)

    yaml.dump(config, open('config.yaml', 'w'))
    hcopy('config.yaml', args.output_dir)

    main(args, config)