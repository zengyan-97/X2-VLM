import os
import copy
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from dataset import build_tokenizer
from dataset.utils import pre_caption
from dataset.pretrain_dataset import TextMaskingGenerator


class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_rpath, max_words=30, prompt=''):
        self.annotation = []
        for f in ann_rpath:
            self.annotation += json.load(open(f, 'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class coco_karpathy_train_mlm(Dataset):
    """
    To train a MLM-based generator
    """
    def __init__(self, transform, image_root, ann_rpath, config):

        self.apply_FG_free = config['apply_FG_free']

        self.annotation = []
        for f in ann_rpath:
            self.annotation += json.load(open(f, 'r'))

        self.transform = transform
        self.image_root = image_root

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.prompt = config['prompt'].strip()

        print("### Always add cls and eos to text tokens")
        self.tokenizer = build_tokenizer(config['text_encoder'])

        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.mask_token = self.tokenizer.mask_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        self.mask_generator = TextMaskingGenerator(self.tokenizer, config['mask_prob'],
                                                   config['max_masks'], config['skipgram_prb'],
                                                   config['skipgram_size'], config['mask_whole_word'])

        self.PAD_mask = self.tokenizer.cls_token_id  # model loss will ignore this
        self.max_tokens = config['max_tokens']
        self.max_masks = config['max_masks']
        self.max_words = config['max_words']

    def __len__(self):
        return len(self.annotation)

    def preprocess(self, text):
        """
        From: UniLM
        MLM For Generative Tasks
        """
        if len(self.prompt):
            prompt_tokens = self.tokenizer.tokenize(self.prompt)
        else:
            prompt_tokens = []

        tokens = self.tokenizer.tokenize(text)
        assert len(tokens) > 0, "len(tokens) <= 0"
        tokens = [self.cls_token] + prompt_tokens + tokens + [self.eos_token]
        tokens = tokens[:self.max_tokens]

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens), num_source_tokens=len(prompt_tokens))

        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_tokens = len(text_ids_masked)
        n_pad = self.max_tokens - n_tokens
        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        text_atts = torch.tril(torch.ones((self.max_tokens, self.max_tokens), dtype=torch.long))

        n_mask = len(masked_pos)
        n_pad = self.max_masks - n_mask
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad
        masked_weight = [1] * n_mask + [0] * n_pad

        position_ids = list(range(len(text_ids_masked)))

        return text_ids_masked, text_atts, position_ids, masked_pos, masked_ids, masked_weight

    def preprocess_fg_free(self, text):
        """
        From: An Investigation of Suitability of Pre-Trained Language Models for Dialogue Generation - Avoiding Discrepancies
        MLM for generative tasks, decreasing finetune-generation discrepancy
        """
        # get tokens
        if len(self.prompt):
            prompt_tokens = self.tokenizer.tokenize(self.prompt)
        else:
            prompt_tokens = []

        tokens = self.tokenizer.tokenize(text)
        assert len(tokens) > 0, "len(tokens) <= 0"
        tokens = [self.cls_token] + prompt_tokens + tokens + [self.eos_token]
        tokens = tokens[:self.max_tokens]

        # do mask
        _, masked_pos_ = self.mask_generator(copy.deepcopy(tokens), num_source_tokens=len(prompt_tokens))

        # prepend [MASK] before "token"
        masked_pos_ = set(masked_pos_)
        tokens_masked = []
        position_ids = []
        masked_pos = []
        masked_ids = []
        i = -1
        for p, t in enumerate(tokens):
            i += 1
            if p in masked_pos_:
                masked_pos.append(len(tokens_masked))
                tokens_masked.append(self.mask_token)
                tokens_masked.append(t)

                position_ids.extend([i, i])

                masked_ids.append(self.tokenizer.convert_tokens_to_ids(t))

            else:
                tokens_masked.append(t)
                position_ids.append(i)

        assert len(tokens_masked) == len(position_ids)
        assert len(masked_pos) == len(masked_ids)

        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int

        max_tokens = self.max_tokens + self.max_masks

        # get attention mask
        text_atts = torch.tril(torch.ones((max_tokens, max_tokens), dtype=torch.long))
        for p in masked_pos:
            text_atts[:, p].fill_(0)
            text_atts[p, p].fill_(1)

        # do pad
        n_tokens = len(text_ids_masked)
        n_pad = max_tokens - n_tokens
        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        position_ids = position_ids + list(range(i+1, i+1+(max_tokens-len(position_ids))))

        n_mask = len(masked_ids)
        n_pad = self.max_masks - n_mask
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad
        masked_weight = [1] * n_mask + [0] * n_pad

        return text_ids_masked, text_atts, position_ids, masked_pos, masked_ids, masked_weight

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        preprocess = self.preprocess_fg_free if self.apply_FG_free else self.preprocess

        text_ids_masked, attention_mask, position_ids, masked_pos, masked_ids, masked_weight = preprocess(ann['caption'])

        return image, text_ids_masked, attention_mask, position_ids, masked_pos, masked_ids, masked_weight

    def collate_fn(self, batch):
        batch_tensors = []
        for x in zip(*batch):
            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors


class coco_karpathy_train_scst(Dataset):
    def __init__(self, transform, image_root, ann_rpath, max_words=30, prompt=''):
        self.annotation = []
        self.image_captions_map = {}

        for f in ann_rpath:
            for ann in json.load(open(f, 'r')):
                self.annotation.append(ann)

                if ann['image'] in self.image_captions_map.keys():
                    self.image_captions_map[ann['image']].append(ann['caption'])
                else:
                    self.image_captions_map[ann['image']] = [ann['caption']]

        counter = Counter()
        for _, v in self.image_captions_map.items():
            counter[len(v)] += 1
        print("### image_captions_map, ", counter, flush=True)

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # w/o prompt
        captions_gt = [pre_caption(c, self.max_words) for c in self.image_captions_map[ann['image']]]

        return image, random.sample(captions_gt, 5)

    def collate_fn(self, batch_sample):
        batch = []
        for x in zip(*batch_sample):
            batch.append(x)

        image_list, captions_gt_list = batch

        images = torch.stack(image_list)

        return images, captions_gt_list


class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_rpath, split):
        self.annotation = json.load(open(ann_rpath, 'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)
