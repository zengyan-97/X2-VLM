import json
import io
import os
import torch
from base64 import b64decode
from torch.utils.data import Dataset
import random
import traceback

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption, sample_frame_ids


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root='', max_words=30,
                 index_key='image_id', vision_key='image', text_key='caption',
                 is_video=False, frame_len=1):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        self.index_key = index_key
        self.vision_key = vision_key
        self.text_key = text_key
        self.is_video = is_video
        self.frame_len = frame_len
        self.training = True

        n = 0
        for ann in self.ann:
            img_id = ann[self.index_key]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        assert isinstance(ann, dict)

        vision_rpath = os.path.join(self.image_root, ann[self.vision_key]) if len(self.image_root) else ann[self.vision_key]

        if self.is_video:
            frames_b64 = json.load(open(vision_rpath, 'r'))

            selected_indices = sample_frame_ids(len(frames_b64), self.frame_len, self.training)

            vision_input = []
            for i in selected_indices:
                image = Image.open(io.BytesIO(b64decode(frames_b64[i]))).convert("RGB")
                image = self.transform(image)
                vision_input.append(image)

        else:
            image = Image.open(vision_rpath).convert('RGB')
            vision_input = self.transform(image)

        caption = pre_caption(ann[self.text_key], self.max_words)

        return vision_input, caption, self.img_ids[ann[self.index_key]]

    def collate_fn(self, batch):
        batch_tensors = []
        for i, x in enumerate(zip(*batch)):
            if x[0] is None:
                batch_tensors.append(None)

            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))

            elif isinstance(x[0], list):
                assert i == 0  # # frames !!! always first
                batch_size = len(x)
                frames = torch.stack(sum(x, []))  # flatten
                _, c, h, w = frames.shape
                frames = frames.reshape([batch_size, self.frame_len, c, h, w])
                batch_tensors.append(frames)

            elif isinstance(x[0], str):  # should be texts, put in tokenizer afterwards
                batch_tensors.append(x)

            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30,
                 index_key='image_id', vision_key='image', text_key='caption',
                 is_video=False, frame_len=1, ):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.index_key = index_key
        self.vision_key = vision_key
        self.text_key = text_key
        self.is_video = is_video
        self.frame_len = frame_len
        self.training = False

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann[self.vision_key])
            self.img2txt[img_id] = []

            assert isinstance(ann[self.text_key], list)

            for i, caption in enumerate(ann[self.text_key]):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        if len(self.image_root):
            image_path = os.path.join(self.image_root, self.ann[index][self.vision_key])
        else:
            image_path = self.ann[index][self.vision_key]

        if self.is_video:
            frames_b64 = json.load(open(image_path, 'r'))
            selected_indices = sample_frame_ids(len(frames_b64), self.frame_len, self.training)

            frames = []
            for i in selected_indices:
                image = Image.open(io.BytesIO(b64decode(frames_b64[i]))).convert("RGB")
                image = self.transform(image)
                frames.append(image)

            frames = torch.stack(frames, dim=0)  # (frame_len, 3, 384, 384)

            return frames, index

        else:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            return image, index