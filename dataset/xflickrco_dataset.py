# Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training (https://arxiv.org/abs/2206.00621)
# Github: https://github.com/zengyan-97/CCLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class xflickrco_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=80):
        self.ann = []
        for f in ann_file:
            for line in open(f):
                ann = json.loads(line)
                for i in range(len(ann['sentences'])):
                    self.ann.append({
                        'caption': ann['sentences'][i],
                        'id': ann['id'],
                        'img_path': ann['img_path']})
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['img_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['id']]


class xflickrco_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=80):
        self.ann = []
        for line in open(ann_file, 'r'):
            ann = json.loads(line)

            # judge if caption if empty
            empty = True
            for sent in ann['sentences']:
                if sent:
                    empty = False
                    break
            if empty:
                print(ann_file, 'has a empty caption')
                continue

            self.ann.append(ann)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['img_path'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['sentences']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        if 'COCO' in self.image[index]:
            image_path = os.path.join(self.image_root['coco'], self.image[index])
        else:
            image_path = os.path.join(self.image_root['flickr30k'], self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
