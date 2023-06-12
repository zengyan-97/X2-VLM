import random
import re
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pickle

import utils
from tqdm import tqdm

from utils.hdfs_io import hexists, hcopy, hopen
from vqaTools.vqaEval import VQAEval
from refTools.evaluation.refEvaluation import RefEvaluation


def sample_clip_ids(clips, mininum_frames: int, clip_captions=None, skip_caption_set=None):
    def _count_frames(index_list):
        n = 0
        for i in index_list:
            if clip_captions is not None:
                if clip_captions[i] in skip_caption_set:
                    continue
            n += len(clips[i])
        return n

    assert isinstance(clips, list) and isinstance(clips[0], list)
    if clip_captions is not None:
        assert isinstance(clip_captions, list) and (len(clip_captions) == len(clips))
        clip_captions = [c.strip() for c in clip_captions]

    if skip_caption_set is None:
        skip_caption_set = {}

    ids = []
    mid = random.choice(range(len(clips)))
    ids.append(mid)

    left = right = mid

    while (_count_frames(ids) < mininum_frames) and (len(ids) < len(clips)):
        if left - 1 < 0:
            i = right + 1
            right = i
        elif right + 1 >= len(clips):
            i = left - 1
            left = i
        else:
            if random.random() < 0.5:
                i = right + 1
                right = i
            else:
                i = left - 1
                left = i

        ids.append(i)

    ids = sorted(ids)
    if clip_captions is not None:
        ids = [i for i in ids if clip_captions[i] not in skip_caption_set]
    return ids


def sample_frame_ids(num_frames: int, num_segments: int, training: bool):
    nf = num_frames
    ns = num_segments  # keep
    out_indices = np.zeros(ns)
    if nf <= ns:
        if training:
            out_indices = np.concatenate((np.arange(nf), np.random.randint(nf, size=ns - nf)), axis=0)
        else:
            out_indices = np.array([(idx % nf) for idx in range(ns)])
        out_indices = np.sort(np.array(out_indices))
    else:
        stride = nf // ns  # at least 1
        strides = np.array([stride] * ns)
        offsets = np.array([1] * (nf - ns * stride) + [0] * (ns * (stride + 1) - nf))
        if training:
            np.random.shuffle(offsets)
        strides += offsets
        cursor = 0
        for idx, each_stride in enumerate(strides):
            left, right = cursor, cursor + each_stride
            cursor += each_stride
            if training:
                out_indices[idx] = np.random.randint(left, right)
            else:
                out_indices[idx] = left

    return [int(i) for i in out_indices]


def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption_raw = caption
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

    return caption


def vqa_eval(vqa, result_file, test_ques_path):
    vqaRes = vqa.loadRes(result_file, test_ques_path)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    # evaluate results
    vqaEval.evaluate()

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    return vqaEval


def write_jsonl(result: list, wpath: str):
    if wpath.startswith('hdfs'):
        with hopen(wpath, 'w') as f:
            for res in result:
                to_write = json.dumps(res, ensure_ascii=False) + '\n'
                f.write(to_write.encode())
    else:
        with open(wpath, 'wt') as f:
            for res in result:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')


def read_jsonl(rpath: str):
    result = []
    if rpath.startswith('hdfs'):
        with hopen(rpath, 'r') as f:
            for line in f:
                result.append(json.loads(line.decode().strip()))
    else:
        with open(rpath, 'rt') as f:
            for line in f:
                result.append(json.loads(line.strip()))

    return result

def collect_result_pkl(result, filename, local_wdir, hdfs_wdir, save_result=False):
    assert isinstance(result, list)
    with open(os.path.join(local_wdir, '%s_rank%d.pkl' % (filename, utils.get_rank())), 'wb') as f:
        pickle.dump(result, f)
    dist.barrier()

    result = []
    final_result_file = ''
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            with open(os.path.join(local_wdir,
                                             '%s_rank%d.pkl' % (filename, rank)), 'rb') as f:
                result += pickle.load(f)

        if save_result:
            final_result_file = os.path.join(local_wdir, '%s.pkl' % filename)
            with open(final_result_file, 'wb') as f:
                pickle.dump(result, f)
            print('result file saved to %s' % final_result_file)
            hcopy(final_result_file, os.path.join(hdfs_wdir, '%s.pkl' % filename))
            print('result file saved to %s' % os.path.join(hdfs_wdir, '%s.pkl' % filename))

    dist.barrier()

    return final_result_file, result if save_result else result

def collect_result(result, filename, local_wdir, hdfs_wdir, write_to_hdfs=False, save_result=False, remove_duplicate='', do_not_collect=False):
    assert isinstance(result, list)
    write_jsonl(result, os.path.join(hdfs_wdir if write_to_hdfs else local_wdir,
                                    '%s_rank%d.json' % (filename, utils.get_rank())))
    dist.barrier()

    if do_not_collect:
        return None

    result = []
    final_result_file = ''
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            result += read_jsonl(os.path.join(hdfs_wdir if write_to_hdfs else local_wdir,
                                             '%s_rank%d.json' % (filename, rank)))

        if remove_duplicate:  # for evaluating captioning tasks
            result_new = []
            id_list = set()
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.add(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        if save_result:
            final_result_file = os.path.join(local_wdir, '%s.json' % filename)
            json.dump(result, open(final_result_file, 'w'), ensure_ascii=False, indent=4)
            print('result file saved to %s' % final_result_file)
            if write_to_hdfs:
                hcopy(final_result_file, os.path.join(hdfs_wdir, '%s.json' % filename))
                print('result file saved to %s' % os.path.join(hdfs_wdir, '%s.json' % filename))

    dist.barrier()

    return final_result_file if save_result else result


def collect_tensor_result(result, filename, local_wdir, hdfs_wdir, write_to_hdfs=False):
    wpath = os.path.join(local_wdir, '%s_rank%d.pth' % (filename, utils.get_rank()))
    torch.save(result, wpath)
    if write_to_hdfs:
        hcopy(wpath, hdfs_wdir)

    dist.barrier()

    result = []
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            rpath = os.path.join(local_wdir, '%s_rank%d.pth' % (filename, rank))
            if write_to_hdfs:
                hcopy(os.path.join(hdfs_wdir, '%s_rank%d.pth' % (filename, rank)), rpath)

            result += torch.load(rpath)

    dist.barrier()

    return result


def grounding_eval(results, dets, cocos, refer, alpha, mask_size=24):
    correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
    correct_A, correct_B, correct_val = 0, 0, 0
    num_A, num_B, num_val = 0, 0, 0

    for res in tqdm(results):

        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        mask = res['pred'].cuda().view(1, 1, mask_size, mask_size)
        mask = F.interpolate(mask, size=(image['height'], image['width']), mode='bicubic').squeeze()

        # rank detection boxes
        max_score = 0
        for det in dets[str(ref['image_id'])]:
            score = mask[int(det[1]):int(det[1] + det[3]), int(det[0]):int(det[0] + det[2])]
            area = det[2] * det[3]
            score = score.sum() / area ** alpha
            if score > max_score:
                pred_box = det[:4]
                max_score = score

        IoU_det = computeIoU(ref_box, pred_box)

        if ref['split'] == 'testA':
            num_A += 1
            if IoU_det >= 0.5:
                correct_A_d += 1
        elif ref['split'] == 'testB':
            num_B += 1
            if IoU_det >= 0.5:
                correct_B_d += 1
        elif ref['split'] == 'val':
            num_val += 1
            if IoU_det >= 0.5:
                correct_val_d += 1

    eval_result = {'val_d': correct_val_d / num_val, 'testA_d': correct_A_d / num_A, 'testB_d': correct_B_d / num_B}

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


def grounding_eval_vlue(results, test_json, alpha, mask_size=24):
    correct_val_d = 0
    num_val = 0

    ref_id_map = {}
    with open(test_json, 'r') as f:
        for sample in json.load(f):
            ref_id_map[sample['ref_id']] = sample

    for res in tqdm(results):

        ref_id = res['ref_id']

        ref_box = ref_id_map[ref_id]['bbox']
        height = ref_id_map[ref_id]['height']
        width = ref_id_map[ref_id]['width']
        dets = ref_id_map[ref_id]['dets']  # (x, y, w, h)

        mask = res['pred'].cuda().view(1, 1, mask_size, mask_size)
        mask = F.interpolate(mask, size=(height, width), mode='bicubic').squeeze()

        # rank detection boxes
        max_score = 0
        for det in dets:
            score = mask[int(det[1]):int(det[1] + det[3]), int(det[0]):int(det[0] + det[2])]
            area = det[2] * det[3]
            score = score.sum() / area ** alpha
            if score > max_score:
                pred_box = det[:4]
                max_score = score

        IoU_det = computeIoU(ref_box, pred_box)

        num_val += 1
        if IoU_det >= 0.5:
            correct_val_d += 1

    eval_result = {'score': correct_val_d / num_val}

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


def grounding_eval_bbox(results, refer):
    correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
    num_A, num_B, num_val = 0, 0, 0

    for res in tqdm(results):
        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        coord = res['pred'].cuda()
        coord[0::2] *= image['width']
        coord[1::2] *= image['height']

        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2

        IoU_det = computeIoU(ref_box, coord)

        if ref['split'] == 'testA':
            num_A += 1
            if IoU_det >= 0.5:
                correct_A_d += 1
        elif ref['split'] == 'testB':
            num_B += 1
            if IoU_det >= 0.5:
                correct_B_d += 1
        elif ref['split'] == 'val':
            num_val += 1
            if IoU_det >= 0.5:
                correct_val_d += 1

    eval_result = {'val_d': correct_val_d / num_val, 'testA_d': correct_A_d / num_A, 'testB_d': correct_B_d / num_B}

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


def grounding_eval_bbox_vlue(results, test_json):
    correct_val_d = 0
    num_val = 0

    ref_id_map = {}
    with open(test_json, 'r') as f:
        for sample in json.load(f):
            ref_id_map[sample['ref_id']] = sample

    for res in tqdm(results):
        ref_id = res['ref_id']

        ref_box = ref_id_map[ref_id]['bbox']
        height = ref_id_map[ref_id]['height']
        width = ref_id_map[ref_id]['width']

        coord = res['pred'].cuda()
        coord[0::2] *= width
        coord[1::2] *= height

        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2

        IoU_det = computeIoU(ref_box, coord)

        num_val += 1
        if IoU_det >= 0.5:
            correct_val_d += 1

    eval_result = {'score': correct_val_d / num_val}

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union


from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def coco_caption_eval(annotation_file, results_file):
    assert os.path.exists(annotation_file)

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}', flush=True)

    return coco_eval



