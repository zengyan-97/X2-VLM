# Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training (https://arxiv.org/abs/2206.00621)
# Github: https://github.com/zengyan-97/CCLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import json
import os
from collections import OrderedDict

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
import base64
import io
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class wit_train_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=80):
        self.ann = []
        for f in ann_file:
            for line in tqdm(open(f)):
                ann = json.loads(line)
                if not ann['caption_reference_description']:
                    continue
                self.ann.append(ann)
        self.transform = transform
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_url']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_str = base64.b64decode(ann['image_content'])
        image = Image.open(io.BytesIO(image_str)).convert("RGB")
        image = self.transform(image)
        try:
            caption = pre_caption(ann['caption_reference_description'], self.max_words)
        except Exception:
            caption = ann['caption_reference_description']

        return image, caption, self.img_ids[ann['image_url']]


class wit_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=80):
        self.ann = []
        for line in open(ann_file, 'r'):
            ann = json.loads(line)
            if not ann['caption_reference_description']:
                continue
            self.ann.append(ann)
        self.transform = transform
        self.max_words = max_words
        self.text = []
        self.image = OrderedDict()
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        img_id = 0
        for ann in self.ann:
            if ann['image_url'] in self.image:
                cur_img_id = self.image[ann['image_url']][0]
                self.img2txt[cur_img_id].append(txt_id)
                self.txt2img[txt_id] = cur_img_id
            else:
                self.img2txt[img_id] = [txt_id]
                self.image[ann['image_url']] = (img_id, ann['image_content'])
                self.txt2img[txt_id] = img_id
                img_id += 1
            if ann['caption_reference_description'] == '.':
                self.text.append(ann['caption_reference_description'])
            else:
                self.text.append(pre_caption(ann['caption_reference_description'], self.max_words))
            txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_str = base64.b64decode(list(self.image.values())[index][1])
        image = Image.open(io.BytesIO(image_str)).convert("RGB")
        image = self.transform(image)

        return image, index
