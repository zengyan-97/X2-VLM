import os
import io
from base64 import b64decode
import json
import random
import csv
from random import random as rand
import torch

from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question, sample_frame_ids

from torchvision.transforms.functional import hflip
from dataset import build_tokenizer


class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root='', vg_root='', split="train", max_ques_words=30, answer_list='',
                 text_encoder='', vision_key='image', q_key='question', ans_key='answer', index_key='question_id', is_video=False, frame_len=1):

        self.careful_hflip = True

        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words

        self.vision_key = vision_key
        self.q_key = q_key
        self.ans_key = ans_key
        self.index_key = index_key
        self.is_video = is_video
        self.frame_len = frame_len

        tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token

        self.training = True

        if split == 'test':
            self.training = False
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))
        
    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def load_image(self, image, is_image_rpath=True, do_hflip=True):
        if is_image_rpath:  # image read path
            image = Image.open(image).convert('RGB')

        else:  # base64 encoding
            # if reading from HDFS, use this:
            image = Image.open(io.BytesIO(b64decode(image))).convert("RGB")

        if (self.split != 'test') and rand() < 0.5:
            if do_hflip:
                image = hflip(image)

        image = self.transform(image)

        return image

    def __getitem__(self, index):

        ann = self.ann[index]
        assert isinstance(ann, dict)

        if 'dataset' in ann.keys():
            if ann['dataset'] == 'vqa':
                image_path = os.path.join(self.vqa_root, ann[self.vision_key])
            elif ann['dataset'] == 'vg':
                image_path = os.path.join(self.vg_root, ann[self.vision_key])
            elif ann['dataset'] == 'gqa':
                image_path = ann['image']
            else:
                raise ValueError(f'dataset == {ann["dataset"]}')

        else:
            image_path = os.path.join(self.vqa_root, ann[self.vision_key]) if len(self.vqa_root) else ann[self.vision_key]

        do_hflip = False
        if self.split != 'test':
            do_hflip = not (self.careful_hflip and self.left_or_right_in(ann[self.q_key], ann[self.ans_key]))

        if self.is_video:
            frames_b64 = json.load(open(image_path, 'r'))

            selected_indices = sample_frame_ids(len(frames_b64), self.frame_len, self.training)

            image = []
            for i in selected_indices:
                image.append(self.load_image(frames_b64[i], is_image_rpath=False, do_hflip=do_hflip))

            image = torch.stack(image, dim=0)  # (frame_len, 3, 384, 384)

        else:
            image = self.load_image(image_path, is_image_rpath=True, do_hflip=do_hflip)

        if self.split == 'test':
            question = pre_question(ann[self.q_key], self.max_ques_words)
            question_id = ann[self.index_key]
            return image, question, question_id

        elif self.split == 'train':
            question = pre_question(ann[self.q_key], self.max_ques_words)

            if ('dataset' in ann.keys()) and (ann['dataset'] == 'vg'):
                answers = [ann[self.ans_key]]
                weights = [0.5]

            else:
                answer_weight = {}
                for answer in ann[self.ans_key]:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann[self.ans_key])
                    else:
                        answer_weight[answer] = 1 / len(ann[self.ans_key])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            # answers = [answer + self.eos_token for answer in answers]  # fix bug

            return image, question, answers, weights

        else:
            raise NotImplementedError


class msrvtt_qa_dataset(Dataset):
    """
    Following CLIPBERT's experimental settings of MSRVTT-QA (modeling it as a 1500-way classification task)
    Files: train.jsonl val.jsonl test.jsonl train_ans2label.json (1500)
    """
    def __init__(self, ann_file, ans2label_file, transform, img_rdir, split="train", max_ques_words=40,
                 text_encoder='', vision_key='video_id', q_key='question', ans_key='answer', frame_len=5):

        self.careful_hflip = True

        self.split = split
        self.ann = []

        if isinstance(ann_file, str):
            ann_file = [ann_file]
        for f in ann_file:
            with open(f, 'r') as f:
                for line in f:
                    self.ann.append(json.loads(line.strip()))

        self.ans2label = json.load(open(ans2label_file, 'r'))

        self.transform = transform
        self.img_rdir = img_rdir
        self.max_ques_words = max_ques_words

        self.vision_key = vision_key
        self.q_key = q_key
        self.ans_key = ans_key

        self.frame_len = frame_len

        tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token

        self.training = True

        if split != 'train':
            self.training = False
            self.max_ques_words = 50  # do not limit question length during test

    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def load_image(self, image, is_image_rpath=True, do_hflip=True):
        if is_image_rpath:  # image read path
            image = Image.open(image).convert('RGB')

        else:  # base64 encoding
            # if reading from HDFS, use this:
            image = Image.open(io.BytesIO(b64decode(image))).convert("RGB")

        if (self.split == 'train') and rand() < 0.5:
            if do_hflip:
                image = hflip(image)

        image = self.transform(image)

        return image

    def __getitem__(self, index):

        ann = self.ann[index]
        assert isinstance(ann, dict)

        assert ann[self.vision_key].startswith('video')
        video_id = int(ann[self.vision_key][5:])
        image_path = os.path.join(self.img_rdir, f"video_{video_id}.json")

        do_hflip = False
        if self.split == 'train':
            do_hflip = not (self.careful_hflip and self.left_or_right_in(ann[self.q_key], ann[self.ans_key]))

        frames_b64 = json.load(open(image_path, 'r'))
        selected_indices = sample_frame_ids(len(frames_b64), self.frame_len, self.training)

        image = []
        for i in selected_indices:
            image.append(self.load_image(frames_b64[i], is_image_rpath=False, do_hflip=do_hflip))

        image = torch.stack(image, dim=0)  # (frame_len, 3, 384, 384)

        question = pre_question(ann[self.q_key], self.max_ques_words)

        try:
            label = self.ans2label[ann[self.ans_key]]
        except KeyError:
            if self.split == 'train':
                print(f"### {ann[self.ans_key]} not in self.ans2label", flush=True)
            label = -100

        return image, question, label


class msvd_qa_dataset(Dataset):
    """
    Following CLIPBERT's experimental settings of MSVD-QA (modeling it as a 1000-way classification task)
    Files: train.jsonl val.jsonl test.jsonl train_ans2label.json (1000)
    """
    def __init__(self, ann_file, ans2label_file, transform, img_rdir, split="train", max_ques_words=40,
                 text_encoder='', vision_key='video_id', q_key='question', ans_key='answer', frame_len=5):

        self.careful_hflip = True

        self.split = split
        self.ann = []

        if isinstance(ann_file, str):
            ann_file = [ann_file]
        for f in ann_file:
            with open(f, 'r') as f:
                for line in f:
                    self.ann.append(json.loads(line.strip()))

        self.ans2label = json.load(open(ans2label_file, 'r'))

        self.transform = transform
        self.img_rdir = img_rdir
        self.max_ques_words = max_ques_words

        self.vision_key = vision_key
        self.q_key = q_key
        self.ans_key = ans_key

        self.frame_len = frame_len

        tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token

        self.training = True

        if split != 'train':
            self.training = False
            self.max_ques_words = 50  # do not limit question length during test

    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def load_image(self, image, is_image_rpath=True, do_hflip=True):
        if is_image_rpath:  # image read path
            image = Image.open(image).convert('RGB')

        else:  # base64 encoding
            # if reading from HDFS, use this:
            image = Image.open(io.BytesIO(b64decode(image))).convert("RGB")

        if (self.split == 'train') and rand() < 0.5:
            if do_hflip:
                image = hflip(image)

        image = self.transform(image)

        return image

    def __getitem__(self, index):

        ann = self.ann[index]
        assert isinstance(ann, dict)

        # assert ann[self.vision_key].startswith('video')
        video_id = ann[self.vision_key]
        image_path = os.path.join(self.img_rdir, f"{video_id}.json")

        do_hflip = False
        if self.split == 'train':
            do_hflip = not (self.careful_hflip and self.left_or_right_in(ann[self.q_key], ann[self.ans_key]))

        frames_b64 = json.load(open(image_path, 'r'))
        selected_indices = sample_frame_ids(len(frames_b64), self.frame_len, self.training)

        image = []
        for i in selected_indices:
            image.append(self.load_image(frames_b64[i], is_image_rpath=False, do_hflip=do_hflip))

        image = torch.stack(image, dim=0)  # (frame_len, 3, 384, 384)

        question = pre_question(ann[self.q_key], self.max_ques_words)

        try:
            label = self.ans2label[ann[self.ans_key]]
        except KeyError:
            if self.split == 'train':
                print(f"### {ann[self.ans_key]} not in self.ans2label", flush=True)
            label = -100

        return image, question, label


class vqa_classify_dataset(Dataset):
    def __init__(self, ann_file, ans2label_file, transform, vqa_root='', vg_root='', split="train", max_ques_words=30, answer_list='',
                 text_encoder='', vision_key='image', q_key='question', ans_key='answer', index_key='question_id', is_video=False, frame_len=1):

        self.careful_hflip = True

        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.ans2label = json.load(open(ans2label_file, 'r'))
        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words

        self.vision_key = vision_key
        self.q_key = q_key
        self.ans_key = ans_key
        self.index_key = index_key
        self.is_video = is_video
        self.frame_len = frame_len

        tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token

        self.training = True

        if split == 'test':
            self.training = False
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))
        
    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def load_image(self, image, is_image_rpath=True, do_hflip=True):
        if is_image_rpath:  # image read path
            image = Image.open(image).convert('RGB')

        else:  # base64 encoding
            # if reading from HDFS, use this:
            image = Image.open(io.BytesIO(b64decode(image))).convert("RGB")

        if (self.split != 'test') and rand() < 0.5:
            if do_hflip:
                image = hflip(image)

        image = self.transform(image)

        return image

    def __getitem__(self, index):

        ann = self.ann[index]
        assert isinstance(ann, dict)

        if 'dataset' in ann.keys():
            if ann['dataset'] == 'vqa':
                image_path = os.path.join(self.vqa_root, ann[self.vision_key])
            elif ann['dataset'] == 'vg':
                image_path = os.path.join(self.vg_root, ann[self.vision_key])
            elif ann['dataset'] == 'gqa':
                image_path = ann['image']
            else:
                raise ValueError(f'dataset == {ann["dataset"]}')

        else:
            image_path = os.path.join(self.vqa_root, ann[self.vision_key]) if len(self.vqa_root) else ann[self.vision_key]

        do_hflip = False
        if self.split != 'test':
            do_hflip = not (self.careful_hflip and self.left_or_right_in(ann[self.q_key], ann[self.ans_key]))

        if self.is_video:
            frames_b64 = json.load(open(image_path, 'r'))

            selected_indices = sample_frame_ids(len(frames_b64), self.frame_len, self.training)

            image = []
            for i in selected_indices:
                image.append(self.load_image(frames_b64[i], is_image_rpath=False, do_hflip=do_hflip))

            image = torch.stack(image, dim=0)  # (frame_len, 3, 384, 384)

        else:
            image = self.load_image(image_path, is_image_rpath=True, do_hflip=do_hflip)

        if self.split == 'test':
            question = pre_question(ann[self.q_key], self.max_ques_words)
            question_id = ann[self.index_key]
            return image, question, question_id

        elif self.split == 'train':
            question = pre_question(ann[self.q_key], self.max_ques_words)

            if ('dataset' in ann.keys()) and (ann['dataset'] == 'vg'):
                answers_pred = [0]*len(self.ans2label.keys())
                try:
                    answers = [self.ans2label[ann[self.ans_key]]] 
                    answers_pred[self.ans2label[ann[self.ans_key]]] += 1
                except KeyError:
                    answers = [-100]
                weights = [0.5]

            else:
                answer_weight = {}
                for answer in ann[self.ans_key]:
                    if answer in self.ans2label.keys():
                        if answer in answer_weight.keys():
                            answer_weight[answer] += 1 / len(ann[self.ans_key])
                        else:
                            answer_weight[answer] = 1 / len(ann[self.ans_key])

                answers = []
                answers_pred = [0]*len(self.ans2label.keys())
                weights = []
                for k,w in zip(list(answer_weight.keys()),list(answer_weight.values())):
                    if str(k) in self.ans2label.keys():
                        answers.append(self.ans2label[str(k)])
                        weights.append(w)
                        answers_pred[self.ans2label[str(k)]] += 1
                
            return image, question, answers, weights, answers_pred

        else:
            raise NotImplementedError


class xgqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root=None, split="train", max_ques_words=30, answer_list='',
                 text_encoder=''):

        self.careful_hflip = True

        self.split = split
        self.ann = []

        if isinstance(ann_file, str):
            ann_file = [ann_file]
        elif not isinstance(ann_file, list):
            raise ValueError

        for f in ann_file:
            ann = json.load(open(f, 'r'))
            if isinstance(ann, list):
                self.ann += ann
            elif isinstance(ann, dict):
                # test set & few-shot train set
                for k, v in ann.items():
                    v['question_id'] = k
                    v['img_id'] = v.pop('imageId')
                    v['sent'] = v.pop('question')

                    # few-shot train set
                    if split == 'train':
                        v['label'] = {v['answer']: 0}

                    self.ann.append(v)

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words

        tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.sep_token

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = list(json.load(open(answer_list, 'r')).keys())

    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.vqa_root, ann['img_id']+'.jpg')

        image = Image.open(image_path).convert('RGB')

        if (self.split != 'test') and rand() < 0.5:
            if self.careful_hflip and self.left_or_right_in(ann['sent'], list(ann['label'].keys())):
                pass
            else:
                image = hflip(image)

        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['sent'], self.max_ques_words)
            question_id = int(ann['question_id'])
            return image, question, question_id

        elif self.split == 'train':
            question = pre_question(ann['sent'], self.max_ques_words)

            answer_weight = {}
            for answer in ann['label'].keys():
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann['label'])
                else:
                    answer_weight[answer] = 1 / len(ann['label'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            # answers = [answer + self.eos_token for answer in answers]  # fix bug

            return image, question, answers, weights

        else:
            raise NotImplementedError

class next_qa_mc_dataset(Dataset):
    """
    Following NExTQA's experimental settings on multiple-choice (modeling it as a 4-option multiple-choice setting)
    Files: train.csv val.csv test.csv 
    """
    def __init__(self, ann_file, transform, img_rdir, split="train", max_ques_words=40,
                 text_encoder='', vision_key='video', q_key='question', ans_key='answer', frame_len=5):

        self.careful_hflip = True

        self.split = split
        self.ann = []

        if isinstance(ann_file, str):
            ann_file = [ann_file]
        # for f in ann_file:
        #     with open(f, 'r') as f:
        #         for line in f:
        #             self.ann.append(json.loads(line.strip()))
        for file_ in ann_file:
            with open(file_,'rt') as f:
                cr = csv.DictReader(f)
                for i,row in enumerate(cr):
                    item_ = {}
                    # train.append(row)
                    for k,v in row.items():
                        item_[k] = v
                    self.ann.append(item_)
                    # print(item_)

        self.transform = transform
        self.img_rdir = img_rdir
        self.max_ques_words = max_ques_words

        self.vision_key = vision_key
        self.q_key = q_key
        self.ans_key = ans_key

        self.frame_len = frame_len

        tokenizer = build_tokenizer(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token

        self.training = True

        if split != 'train':
            self.training = False
            self.max_ques_words = 50  # do not limit question length during test

    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def load_image(self, image, is_image_rpath=True, do_hflip=True):
        if is_image_rpath:  # image read path
            image = Image.open(image).convert('RGB')

        else:  # base64 encoding
            # if reading from HDFS, use this:
            image = Image.open(io.BytesIO(b64decode(image))).convert("RGB")

        if (self.split == 'train') and rand() < 0.5:
            if do_hflip:
                image = hflip(image)

        image = self.transform(image)

        return image

    def __getitem__(self, index):

        ann = self.ann[index]
        assert isinstance(ann, dict)

        video_id = int(ann[self.vision_key])
        image_path = os.path.join(self.img_rdir, f"{video_id}.json")

        do_hflip = False
        if self.split == 'train':
            do_hflip = not (self.careful_hflip and self.left_or_right_in(ann[self.q_key], ann[self.ans_key]))

        frames_b64 = json.load(open(image_path, 'r'))
        selected_indices = sample_frame_ids(len(frames_b64), self.frame_len, self.training)

        image = []
        for i in selected_indices:
            image.append(self.load_image(frames_b64[i], is_image_rpath=False, do_hflip=do_hflip))

        image = torch.stack(image, dim=0)  # (frame_len, 3, 384, 384)
        
        question = pre_question(ann[self.q_key], self.max_ques_words)

        cand_list = [ann["a0"], ann["a1"], ann["a2"], ann["a3"], ann["a4"]]

        label = ann[self.ans_key]

        return image, question, label, cand_list