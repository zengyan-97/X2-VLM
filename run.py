# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import sys
import time
import random
import argparse

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

############ Set it correctly for distributed training across nodes
NNODES = int(os.getenv("ARNOLD_WORKER_NUM"))  # e.g. 1/2/3/4
NPROC_PER_NODE = int(os.getenv("ARNOLD_WORKER_GPU"))  # e.g. 8

MASTER_ADDR = os.getenv("METIS_WORKER_0_HOST")
MASTER_PORT = int(os.getenv("METIS_WORKER_0_PORT"))
NODE_RANK = int(os.getenv("ARNOLD_ID"))  # e.g. 0/1/2
############

print("NNODES, ", NNODES)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NODE_RANK, ", NODE_RANK)


def get_nnodes(args):  # when using only part of nodes
    if args.dist == 'all':
        return NNODES

    elif args.dist == '2':
        assert NNODES >= 2
        return 2

    else:
        return 1


def get_dist_launch(args):  # some examples
    if args.dist == 'all':  # use all nodes
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '2':
        assert int(os.getenv("ARNOLD_WORKER_NUM")) >= 2
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=2 --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '1':
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=1 --master_port={:}".format(NPROC_PER_NODE, args.master_port)

    elif args.dist == 'f4f2':
        return "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 python3 -m torch.distributed.launch --nproc_per_node=2 " \
               "--nnodes=1 --master_port={:}".format(args.master_port)

    elif args.dist == 'f4l2':
        return "CUDA_VISIBLE_DEVICES=2,3 WORLD_SIZE=2 python3 -m torch.distributed.launch --nproc_per_node=2 " \
               "--nnodes=1 --master_port={:}".format(args.master_port)

    elif args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.launch --nproc_per_node=4 " \
               "--nnodes=1 --master_port={:}".format(args.master_port)

    elif args.dist == 'l4':
        return "CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=4 python3 -m torch.distributed.launch --master_port=12345 --nproc_per_node=4 " \
               "--nnodes=1 --master_port={:}".format(args.master_port)

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
               "--nnodes=1 --master_port={:}".format(num, args.master_port)

    else:
        raise ValueError


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    assert hexists(file_hdfs)
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]
        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_pretrain(args):
    dist_launch = get_dist_launch(args)

    use_env = 'Pretrain.py'

    print(f"### Start pre-training {use_env}", flush=True)
    os.system(f"{dist_launch} --use_env {use_env} --seed {args.seed} "
              f"--epoch {args.epoch} --config {args.config} --output_dir {args.output_dir} "
              f"{f'--checkpoint {args.checkpoint}' if args.checkpoint else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} " )


def run_nlvr2(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/nlvr2")

    print("### Training NLVR2", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env NLVR.py --config {args.config} "
              f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} "
              f"{'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} " )


def run_itr_flickr(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/flickr30k-images")

    print("### Training Retrieval Flickr", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env Retrieval.py --config {args.config} {'--pick_best_r1' if args.pick_best_r1 else ''} "
              f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_itr_coco(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    print("### Training Retrieval COCO", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env Retrieval.py --config {args.config} "
              f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} {'--pick_best_r1' if args.pick_best_r1 else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_itr_msrvtt(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/msrvtt")

    print("### Training Retrieval MSR-VTT", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env Retrieval.py --pick_best_t2v --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} --k_test {args.k_test} {'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_vqa(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco") and os.path.exists("images/visualgenome")

    print("### Training VQA", flush=True)

    if not os.path.exists(os.path.join(args.output_dir, 'result')):
        os.mkdir(os.path.join(args.output_dir, 'result'))

    os.system(f"{dist_launch} "
              f"--use_env VQA.py --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_vqa_msrvtt(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/msrvtt")

    print("### Training VQA MSR-VTT", flush=True)

    if not hexists(os.path.join(args.output_dir, 'result')):
        hmkdir(os.path.join(args.output_dir, 'result'))

    os.system(f"{dist_launch} "
              f"--use_env VQA_msrvtt.py --config {args.config} "
              f"--output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_vqa_msvd(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/msvd")

    print("### Training VQA MSVD", flush=True)
    if not os.path.exists(args.config): args.config = f'./configs/{args.model}/VQA_msvd.yaml'

    if not hexists(os.path.join(args.output_dir, 'result')):
        hmkdir(os.path.join(args.output_dir, 'result'))

    os.system(f"{dist_launch} "
              f"--use_env VQA_msvd.py --config {args.config} "
              f"--output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_refcoco(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    print("### Training RefCOCO with bbox", flush=True)
    if not os.path.exists(args.config): args.config = f"configs/{args.model}/Grounding_bbox.yaml"

    os.system(f"{dist_launch} "
              f"--use_env Grounding_bbox.py --config {args.config} "
              f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} "
              f"{'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_coco_captioning_mlm(args):
    """
    Follow BEiT-3, use UniLM way to generate captions
    (MLM for generative tasks)
    """
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    print("### Training COCO Captioning (MLM)", flush=True)

    if not os.path.exists(args.config):
        args.config = f'./configs/{args.model}/Captioning_MLM.yaml'

    os.system(f"{dist_launch} "
              f"--use_env Captioning_MLM.py --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{f'--override_cfg {args.override_cfg}' if len(args.override_cfg) else ''} ")


def run_itr_flickr_mm(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/flickr30k-images")

    print("### Training Retrieval Flickr", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env 'XRetrieval.py' --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_itr_coco_mm(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    print("### Training Retrieval COCO", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env 'XRetrieval.py' --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_xvnli(args):
    dist_launch = get_dist_launch(args)

    print("### Training xvnli", flush=True)

    assert os.path.exists("images/flickr30k-images")

    evaluate = ' --evaluate' if args.evaluate else ''

    trans_test = ' --gmt' if args.gmt else ''
    os.system(f"{dist_launch} "
              f"--use_env XVNLI.py --config {args.config} "
              f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} "
              f"--lr {args.lr}" + trans_test + evaluate)


def run_xgqa(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/gqa")

    print("### Training XGQA", flush=True)

    os.system(f"{dist_launch} "
              f"--use_env XGQA.py --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{'--load_vqa_pretrain --fewshot ' + args.fewshot if args.fewshot else ''} --lr {args.lr}")


def run_marvl(args):
    dist_launch = get_dist_launch(args)

    if not os.path.exists('data_mm/marvl'):
        from utils.marvl_preproc import marvl_preproc
        marvl_preproc('iglue/datasets/marvl', 'data_mm/marvl')

    assert os.path.exists("images/nlvr2")
    assert os.path.exists("images/marvl_official")
    assert os.path.exists("images/marvl_fewshot")
    assert os.path.exists('data_mm/marvl')

    print("### Training MARVL", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env MARVL.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} "
              f"{'--evaluate' if args.evaluate else ''} "
              f"--lr {args.lr} {'--fewshot ' + args.fewshot if args.fewshot else ''}")


def run_xflickrco(args):
    dist_launch = get_dist_launch(args)

    print("### Training xFlickr&CO", flush=True)

    assert os.path.exists("images/val2014")  # coco
    assert os.path.exists("images/flickr30k-images")

    evaluate = ' --evaluate' if args.evaluate else ''
    trans_test = ' --gmt' if args.gmt else ''
    os.system(f"{dist_launch} "
            f"--use_env xFlickrCO.py --config {args.config} "
            f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
            f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --checkpoint {args.checkpoint} "
            f"--lr {args.lr} " + trans_test + evaluate)


def run_wit(args):
    dist_launch = get_dist_launch(args)

    print("### Training WIT", flush=True)

    assert os.path.exists("data_mm/wit")

    evaluate = ' --evaluate' if args.evaluate else ''

    trans_test = ' --gmt' if args.gmt else ''
    os.system(f"{dist_launch} "
            f"--use_env WIT.py --config {args.config} "
            f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
            f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --checkpoint {args.checkpoint}" + trans_test + evaluate)


def run(args):
    if args.task == 'pretrain_DIY':
        run_pretrain(args)

    elif args.task == 'itr_coco':
        run_itr_coco(args)

    elif args.task == 'itr_flickr':
        run_itr_flickr(args)

    elif args.task == 'itr_coco_msrvtt':
        run_itr_msrvtt(args)

    elif args.task == 'vqa':
        run_vqa(args)

    elif args.task == 'vqa_msrvtt':
        run_vqa_msrvtt(args)

    elif args.task == 'vqa_msvd':
        run_vqa_msvd(args)

    elif args.task == 'nlvr':
        run_nlvr2(args)

    elif args.task == 'refcoco_bbox':
        run_refcoco(args)

    elif args.task == 'coco_captioning_mlm':
        assert os.path.exists("data/finetune/coco_karpathy/coco_karpathy_restval.json")
        run_coco_captioning_mlm(args)

    elif args.task == 'itr_coco_mm':
        run_itr_coco_mm(args)

    elif args.task == 'itr_multi30k_mm':
        run_itr_flickr_mm(args)

    elif args.task == 'xvnli':
        run_xvnli(args)

    elif args.task == 'xgqa':
        run_xgqa(args)

    elif args.task == 'marvl':
        run_marvl(args)

    elif args.task == 'xflickrco':
        run_xflickrco(args)

    elif args.task == 'wit':
        run_wit(args)

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")

    # for --task "DIY"
    parser.add_argument('--use_env', default="", type=str, help="use_env for DIY")
    parser.add_argument('--build_env_from', default="", type=str, help="build env for DIY")

    parser.add_argument('--config', default='', type=str, help="if not given, use default")
    parser.add_argument('--config_dp', default='', type=str, help="if not given, use default")
    parser.add_argument('--model', default='beitB-bertB-ft', type=str, help="to set default fine-tuning configs")

    parser.add_argument('--epoch', default=-1, type=int, help="for pre-training (debug) only")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")

    parser.add_argument('--checkpoint', default='', type=str, help="for domain pretraining or fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")
    parser.add_argument('--load_domain_pretrain', action='store_true', help="evaluation on downstream tasks")

    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, required=True, help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--output_hdfs', type=str, default='', help="HDFS path required by VQA and Refcoco, "
                                                                    "to collect eval results among nodes")

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--master_port', default=12345, type=int)

    parser.add_argument('--k_test', default=-1, type=int, help="for retrieval evaluation, i am trying to find the best one for MSRVTT")
    parser.add_argument('--num_workers', default=-1, type=int, help="num_workers config")

    parser.add_argument('--wait', default=0, type=int, help="wait x minutes before running the code")

    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")

    parser.add_argument('--pick_best_r1', action='store_true', help="for retrieval, save best ckpt by r@1")

    # for multilingual tasks
    parser.add_argument('--fewshot', default='', type=str, help="IGLUE fewshot. <lang>,<shot_num>, eg: ar,25")
    parser.add_argument('--lr', default=0., type=float, help="learning rate")
    parser.add_argument('--gmt', action='store_true', help="whether use google machine translation as test set")

    # KDistill Related
    parser.add_argument('--load_trained', action='store_true', help="whether loading domain-pretrained ckpt; i use this opt to reduce the number of layers")

    args = parser.parse_args()

    if args.fewshot:
        raise NotImplementedError

    start = time.time()

    while True:
        t = int((time.time() - start) / 60)
        if t >= args.wait:
            break
        else:
            print(f"### wait {t} minutes (/{args.wait})", flush=True)
            time.sleep(60)

    if MASTER_ADDR == 'SET_IT':
        print("### warning: the settings for distributed training is not filled (ignore this if you only use one node)")

    if '/SET/PATH/TO/hadoop/bin/hdfs' in HADOOP_BIN:
        print("### warning: you have not set the path to hadoop_bin (ignore this if you don't use HDFS)")

    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)

    if len(args.override_cfg) > 0:
        args.override_cfg = args.override_cfg.strip('\"')
        args.override_cfg = '\"{}\"'.format(args.override_cfg)

    if len(args.output_hdfs):
        assert hexists(os.path.dirname(args.output_hdfs))
        hmkdir(args.output_hdfs)

    assert hexists(args.config)
    if args.config.startswith('hdfs://'):
        args.config = get_from_hdfs(args.config)

    if args.checkpoint.startswith('hdfs://'):
        args.checkpoint = get_from_hdfs(args.checkpoint)

    run(args)

