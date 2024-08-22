# Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training (https://arxiv.org/abs/2206.00621)
# Github: https://github.com/zengyan-97/CCLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class xvnli_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=80):
        self.label_mapper = {"contradiction": 0, "entailment": 1, "neutral": 2}
        
        self.ann = []

        if type(ann_file) == str:
            ann_file = [ann_file]

        invalid_cnt = 0
        for f in ann_file:
            for line in open(f, 'r'):
                ann = json.loads(line)
                if ann['gold_label'] not in self.label_mapper:
                    invalid_cnt += 1
                    continue
                self.ann.append(ann)

        if not self.ann:
            raise ValueError(f"ann_file == {ann_file}")

        print('data num: ', len(self.ann))
        print('invalid num: ', invalid_cnt)
        
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['Flikr30kID']+'.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        sentence = pre_caption(ann['sentence2'], self.max_words)

        label = self.label_mapper[ann['gold_label']]

        return image, sentence, label
