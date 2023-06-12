import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class nlvr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root=None):
        self.ann = []

        if isinstance(ann_file, list):
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))

        elif isinstance(ann_file, str):
            self.ann += json.load(open(ann_file, 'r'))

        else:
            raise ValueError(f"ann_file == {ann_file}")

        self.transform = transform
        self.image_root = image_root
        self.max_words = 30
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        if self.image_root is None:
            image0_path = ann['images'][0]
        else:
            image0_path = os.path.join(self.image_root, ann['images'][0])

        image0 = Image.open(image0_path).convert('RGB')
        image0 = self.transform(image0)

        if self.image_root is None:
            image1_path = ann['images'][1]
        else:
            image1_path = os.path.join(self.image_root, ann['images'][1])

        image1 = Image.open(image1_path).convert('RGB')
        image1 = self.transform(image1)

        sentence = pre_caption(ann['sentence'], self.max_words)

        if (ann['label'] == 'True') or (ann['label'] is True):
            label = 1

        elif (ann['label'] == 'False') or (ann['label'] is False):
            label = 0

        else:
            raise ValueError(f"unsupported label: {ann['label']}")

        return image0, image1, sentence, label
