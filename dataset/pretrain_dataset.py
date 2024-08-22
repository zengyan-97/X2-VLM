# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import json
import copy
import math
import random
import sys
import re
import io
import traceback
from base64 import b64decode

from random import randint, shuffle
from random import random as rand

import numpy as np

import torch
from torchvision.transforms.functional import hflip, resize
from torchvision.transforms import InterpolationMode


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset import build_tokenizer
from dataset.utils import pre_caption, sample_frame_ids, sample_clip_ids
from dataset.dist_dataset import DistLineReadingDataset


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True, use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        print("len(tokenizer.id2token), ", len(self.id2token), flush=True)

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list, num_source_tokens=0):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(
            1, int(round((len(tokens)-num_source_tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        if tokens[0] == self.cls_token:
            special_pos = set(range(1+num_source_tokens))  # will not be masked
            cand_pos = list(range(1+num_source_tokens, len(tokens)))
        else:
            special_pos = set(range(num_source_tokens))  # will not be masked
            cand_pos = list(range(num_source_tokens, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos


class ImageTextJsonDataset(DistLineReadingDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None,
                 add_eos=True, is_aux=False, config_key='images'):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        # Dataset Settings
        self.image_key = config[config_key]['image_key']
        self.is_image_rpath = config[config_key]['is_image_rpath']
        self.caption_key = config[config_key]['caption_key']
        if is_aux:
            self.caption_key = config[config_key]['aux_caption_key']

        self.batch_size = config[config_key]['batch_size']
        self.tokenized = config[config_key]['tokenized']
        assert self.tokenized is False, "not implemented"

        if 'language_chosen' in config[config_key].keys():
            assert isinstance(config[config_key]['language_chosen'], str)
            self.language_chosen = config[config_key]['language_chosen']
        else:
            self.language_chosen = None  # pick one randomly

        # Other Settings

        self.transform = transform
        self.image_res = config['image_res']
        self.patch_size = config['patch_size']
        assert self.image_res % self.patch_size == 0
        self.num_patch = int(self.image_res / self.patch_size)

        self.print_broken_data = config.get('print_broken_data', True)
        print("### Always add cls and eos to text tokens")
        self.tokenizer = build_tokenizer(config['text_encoder'])

        self.max_tokens = config[config_key]['max_tokens'] if 'max_tokens' in config[config_key] else config['max_tokens']
        self.max_words = self.max_tokens  # update: 20221216

        if 'mask_prob' in config:  # update: 20221216
            self.cls_token = self.tokenizer.cls_token
            self.eos_token = self.tokenizer.sep_token
            self.pad_token_id = self.tokenizer.pad_token_id
            self.add_eos = True  # update 20220307: consistent with some fine-tuning tasks

            self.mask_token_id = self.tokenizer.mask_token_id
            self.PAD_mask = -100  # loss will ignore this
            self.max_masks = config['max_masks']
            if ('bert-base-uncased' not in config['text_encoder']) and ('bert-large-uncased' not in config['text_encoder']):
                config['mask_whole_word'] = False
                print("### Set mask_whole_word to False", flush=True)
                # assert config['mask_whole_word'] is False, "not implemented"

            self.mask_generator = TextMaskingGenerator(self.tokenizer, config['mask_prob'],
                                                       config['max_masks'], config['skipgram_prb'],
                                                       config['skipgram_size'], config['mask_whole_word'])

    def get_caption(self, caption):
        if isinstance(caption, list):
            caption = random.choice(caption)

        if isinstance(caption, str):
            return caption

        elif isinstance(caption, dict):  # compatible to my multilingual data
            if self.language_chosen is None:
                c = random.choice(list(caption.values()))
                assert isinstance(c, str)
                return c
            else:
                assert isinstance(caption[self.language_chosen], str)
                return caption[self.language_chosen]

        else:
            raise ValueError(caption)

    def load_image(self, data):
        if self.is_image_rpath:  # image read path
            image = Image.open(data).convert('RGB')

        else:  # base64 encoding
            # if reading from HDFS, use this:
            image = Image.open(io.BytesIO(b64decode(data))).convert("RGB")

        return image

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                assert isinstance(ann, dict), "ann is not dict"

                caption = self.get_caption(ann[self.caption_key])

                image = self.load_image(ann[self.image_key])
                image = self.transform(image)

                if not len(caption):
                    del ann[self.image_key]
                    raise ValueError(ann)

                text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

                yield image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

            except Exception as e:
                if self.print_broken_data:
                    print(traceback.format_exc())
                    print('encounter broken data: %s' % e)
                    print('-'*20, flush=True)

    def preprocess(self, text):
        if hasattr(self, 'language_chosen'):
            if self.language_chosen == 'zh':
                text = text.replace(' ', '')

        text = pre_caption(text, self.max_words)
        tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

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


class FrameTextDataset(ImageTextJsonDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None,
                 add_eos=True, config_key='videos', training=True):

        super().__init__(config, data_path, rank=rank, world_size=world_size, shuffle=shuffle,
                         repeat=repeat, transform=transform, config_key=config_key)

        self.frame_key = config[config_key]['image_key']
        self.frame_len = config[config_key]['frame_len']
        self.use_random_sampling = config[config_key].get('use_random_sampling', False)
        self.combine_continuous_clips = config[config_key].get('combine_continuous_clips', False)

        self.mininum_frames_before_sampling = -1
        if self.combine_continuous_clips:
            self.mininum_frames_before_sampling = config[config_key]['mininum_frames_before_sampling']

        self.training = training

        self.skip_caption_set = {'[Music]'}

    def get_frames(self, frames):
        assert isinstance(frames, list)
        selected_indices = sample_frame_ids(len(frames), self.frame_len, self.training)

        selected_frames = [frames[i] for i in selected_indices]
        if len(selected_frames) != self.frame_len:
            raise ValueError('video only has %s frames' % (len(frames)))

        return selected_frames

    def get_clips(self, clips, clip_captions, is_continuous=False):
        """
        Returns:
            frames: list of b64
            clip ids: list of int
        """
        assert isinstance(clips, list) and isinstance(clips[0], list)
        assert isinstance(clip_captions, list)

        if len(clips) == 1:
            return clips[0], [0]
        else:
            if is_continuous:
                ids = sample_clip_ids(clips, mininum_frames=self.mininum_frames_before_sampling,
                                      clip_captions=clip_captions, skip_caption_set=self.skip_caption_set)
                frames = []
                for i in ids:
                    frames += clips[i]
                return frames, ids

            else:
                i = random.choice(range(len(clips)))
                while clip_captions[i] in self.skip_caption_set:  # for Howto100M
                    i = random.choice(range(len(clips)))

                return clips[i], [i]

    def get_vision_input(self, ann):

        video = ann[self.frame_key]

        assert isinstance(video, list)

        if isinstance(video[0], list):
            is_continuous = ann['is_continuous']
            clip_frames, clip_ids = self.get_clips(video, ann[self.caption_key], is_continuous=is_continuous)
            return self.get_frames(clip_frames), clip_ids

        else:
            return self.get_frames(video), None

    def get_caption(self, caption, clip_ids):
        if isinstance(caption, list):
            if clip_ids is None:
                caption = random.choice(caption)
            else:
                caption = ' '.join([caption[i] for i in clip_ids])  # english only

        assert isinstance(caption, str)
        return caption

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                assert isinstance(ann, dict), "ann is not dict"

                frames_input, clip_ids = self.get_vision_input(ann)

                caption = self.get_caption(ann[self.caption_key], clip_ids)

                if not len(caption):
                    del ann[self.frame_key]
                    raise ValueError(ann)

                text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

                frames = []
                for frame in frames_input:
                    image = self.load_image(frame)
                    image = self.transform(image)
                    frames.append(image)

                yield frames, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

            except Exception as e:
                if self.print_broken_data:
                    print(traceback.format_exc())
                    print('encounter broken data: %s' % e)
                    print('-'*20, flush=True)

    def collate_fn(self, batch):
        batch_tensors = []
        for i, x in enumerate(zip(*batch)):
            if i == 0:  # frames !!! always first
                assert isinstance(x[0], list)
                batch_size = len(x)
                frames = torch.stack(sum(x, []))  # flatten
                _, c, h, w = frames.shape
                frames = frames.reshape([batch_size, self.frame_len, c, h, w])
                batch_tensors.append(frames)
            else:
                if x[0] is None:
                    batch_tensors.append(None)

                elif isinstance(x[0], torch.Tensor):
                    batch_tensors.append(torch.stack(x))

                elif isinstance(x[0], str):  # should be texts, put in tokenizer afterwards
                    batch_tensors.append(x)

                else:
                    batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors


class RegionTextJsonDataset(ImageTextJsonDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None, box_transform=None, config_key='regions'):
        super().__init__(config, data_path, rank=rank, world_size=world_size, shuffle=shuffle,
                         repeat=repeat, transform=transform, config_key=config_key)
        # Dataset Settings
        assert self.caption_key == 'caption', "please follow my data format"
        self.careful_hflip = config[config_key].get('careful_hflip', False)
        self.max_regions = config[config_key]['max_regions']
        self.min_perc_in_image = config[config_key]['min_perc_in_image']

        self.box_transform = box_transform

    def get_bbox(self, ann):
        x, y, w, h = ann['bb']
        return int(x), int(y), int(w), int(h)

    def left_or_right_in_caption(self, ann):
        def _in_it(elem):
            if isinstance(elem['caption'], list):
                for caption in elem['caption']:
                    if ('left' in caption) or ('right' in caption):
                        return True
            else:
                if ('left' in elem['caption']) or ('right' in elem['caption']):
                    return True

        if 'caption' in ann.keys():
            if _in_it(ann):
                return True

        for elem in ann['elems']:
            if _in_it(elem):
                return True

        return False

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                assert isinstance(ann, dict), "ann is not dict"

                try:
                    image = Image.open(ann[self.image_key]).convert('RGB') if self.is_image_rpath \
                        else Image.open(io.BytesIO(b64decode(ann[self.image_key]))).convert("RGB")
                except Warning:
                    raise ValueError("### Warning: RegionTextJsonDataset Image.open")

                W, H = image.size

                # random crop
                x, y, w, h = self.get_bbox(random.choice(ann['elems']))
                assert (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (h > 0), "elem invalid"

                x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
                x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H), H)
                w0, h0 = x1 - x0, y1 - y0
                assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (h0 > 0), "elem randomcrop, invalid"

                image = image.crop((x0, y0, x0 + w0, y0 + h0))
                W, H = image.size

                do_hflip = False
                if rand() < 0.5:
                    if self.careful_hflip and self.left_or_right_in_caption(ann):
                        pass
                    else:
                        image = hflip(image)
                        do_hflip = True

                image = resize(image, [self.image_res, self.image_res], interpolation=InterpolationMode.BICUBIC)
                image = self.box_transform(image)

                text_ids_list = []
                text_ids_masked_list = []
                text_atts_list = []
                masked_pos_list = []
                masked_ids_list = []
                image_atts_list = []

                target_bbox_list = []
                is_image_list = []

                max_elems = self.max_regions

                if 'caption' in ann.keys():

                    caption = self.get_caption(ann['caption'])

                    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

                    text_ids_list.append(text_ids)
                    text_atts_list.append(text_atts)
                    text_ids_masked_list.append(text_ids_masked)
                    masked_pos_list.append(masked_pos)
                    masked_ids_list.append(masked_ids)

                    image_atts_list.append([1] * (self.num_patch ** 2 + 1))
                    target_bbox_list.append(torch.tensor([0.5, 0.5, 1, 1], dtype=torch.float))
                    is_image_list.append(1)

                    max_elems -= 1

                elems = random.sample(ann['elems'], len(ann['elems']))

                for elem in elems:
                    if max_elems <= 0:
                        break

                    x, y, w, h = self.get_bbox(elem)

                    xx, yy = max(x0, x), max(y0, y)
                    xm, ym = min(x0 + w0, x + w), min(y0 + h0, y + h)
                    if (xm > xx) and (ym > yy):
                        if (xm - xx) * (ym - yy) / (w * h) > self.min_perc_in_image:
                            x, y, w, h = xx, yy, xm - xx, ym - yy  # part inside the cropped image

                            # axis transform: after crop
                            x = x - x0
                            y = y - y0

                            if do_hflip:  # flipped applied
                                x = (W - x) - w  # W is w0

                            # resize applied
                            x = self.image_res / W * x
                            w = self.image_res / W * w
                            y = self.image_res / H * y
                            h = self.image_res / H * h

                            caption = self.get_caption(elem['caption'])

                            if 'attributes' in elem.keys():
                                elem_attr = self.get_caption(elem['attributes'])
                                caption = elem_attr + ' ' + caption

                            text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)
                            image_atts = self.get_image_attns(x, y, w, h)

                            text_ids_list.append(text_ids)
                            text_atts_list.append(text_atts)
                            text_ids_masked_list.append(text_ids_masked)
                            masked_pos_list.append(masked_pos)
                            masked_ids_list.append(masked_ids)
                            image_atts_list.append(image_atts)

                            center_x = x + 1 / 2 * w
                            center_y = y + 1 / 2 * h

                            target_bbox_list.append(torch.tensor([center_x / self.image_res, center_y / self.image_res,
                                                                  w / self.image_res, h / self.image_res],
                                                                 dtype=torch.float))

                            is_image_list.append(0)

                            max_elems -= 1

                image_list = [image] if len(text_ids_list) else []

                yield image_list, text_ids_list, text_atts_list, text_ids_masked_list, masked_pos_list, \
                      masked_ids_list, image_atts_list, target_bbox_list, is_image_list

            except Exception as e:
                if self.print_broken_data:
                    print(traceback.format_exc())
                    print('encounter broken data: %s' % e)
                    print('-' * 20, flush=True)

    def get_image_attns(self, x, y, w, h):
        x_min = min(math.floor(x / self.patch_size), self.num_patch - 1)
        x_max = max(x_min+1, min(math.ceil((x+w) / self.patch_size), self.num_patch))  # exclude

        y_min = min(math.floor(y / self.patch_size), self.num_patch - 1)
        y_max = max(y_min+1, min(math.ceil((y+h) / self.patch_size), self.num_patch))  # exclude

        image_atts = [0] * (1 + self.num_patch ** 2)
        image_atts[0] = 1  # always include [CLS]
        for j in range(x_min, x_max):
            for i in range(y_min, y_max):
                index = self.num_patch * i + j + 1
                assert (index > 0) and (index <= self.num_patch ** 2), f"patch index out of range, index: {index}"
                image_atts[index] = 1

        return image_atts

    def collate_fn(self, batch_sample):
        batch = []
        for x in zip(*batch_sample):
            batch.append(x)

        images, batch = batch[0], batch[1:]

        idx_to_group_img = []
        img_idx = -1
        for sample in batch[0]:
            n_elems = len(sample)
            if n_elems > 0:
                img_idx += 1
                idx_to_group_img.extend([img_idx] * n_elems)  # flatten

        batch_size = self.batch_size
        n_elems = len(idx_to_group_img)
        to_keep = list(range(n_elems))
        if n_elems >= batch_size:
            to_keep = random.sample(to_keep, batch_size)
        else:
            # fixed batch_size is required. otherwise, the process will be blocked. so, i do pad here.
            # but pad causes wrong calculation for contrastive learning.
            # Set appropriate batch_size, max_images, and max_regions to avoid frequent padding.
            try:
                to_pad = random.sample(to_keep, batch_size - n_elems)
                to_keep += to_pad
                print("### warning: pad region_batch by sampling, ", len(to_pad), flush=True)

            except ValueError:
                print("### warning: pad region_batch by expanding, ", batch_size-len(to_keep), flush=True)
                to_keep = (to_keep * math.ceil(batch_size/len(to_keep)))[:batch_size]

        images = torch.stack(sum(images, []))  # flatten
        idx_to_group_img = torch.tensor([idx_to_group_img[index] for index in to_keep], dtype=torch.long)

        batch_tensors = [images, idx_to_group_img]
        for x in [sum(x, []) for x in batch]:

            x = [x[index] for index in to_keep]

            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors


class TextJsonDataset(DistLineReadingDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.print_broken_data = config.get('print_broken_data', True)

        self.text_key = config['texts']['text_key']
        self.batch_size = config['texts']['batch_size']
        self.tokenized = config['texts']['tokenized']
        assert self.tokenized is False, "not implemented"

        self.add_eos = True  # update 20220307: consistent with some fine-tuning tasks
        print("### Always add cls and eos to text tokens")
        self.tokenizer = build_tokenizer(config['text_encoder'])

        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        if ('bert-base-uncased' not in config['text_encoder']) and ('bert-large-uncased' not in config['text_encoder']):
            config['mask_whole_word'] = False
            print("### Set mask_whole_word to False", flush=True)
            # assert config['mask_whole_word'] is False, "not implemented"

        self.mask_generator = TextMaskingGenerator(self.tokenizer, config['texts']['mask_prob'],
                                                   config['texts']['max_masks'], config['skipgram_prb'],
                                                   config['skipgram_size'], config['mask_whole_word'])

        self.PAD_mask = -100  # loss will ignore this
        self.max_words = config['texts']['max_words']
        self.max_tokens = config['texts']['max_tokens']
        self.max_masks = config['texts']['max_masks']
        self.minimum_words = config['texts']['minimum_words']

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                assert isinstance(ann, dict), "ann is not dict"

                text = ann[self.text_key].strip()
                text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(text)

                yield text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

            except Exception as e:
                if self.print_broken_data:
                    print(traceback.format_exc())
                    print('encounter broken data: %s' % e)
                    print('-'*20, flush=True)

    def pre_caption(self, caption, max_words, minimum_words):
        caption_raw = caption
        # caption = re.sub(
        #     r"([,.'!?\"()*#:;~])",
        #     ' ',
        #     caption.lower(),
        # ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ').lower()

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) < minimum_words:
            raise ValueError(f"too short text (raw: {caption_raw})")
        elif len(caption_words) >= max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    def preprocess(self, text):
        if hasattr(self, 'language_chosen'):
            if self.language_chosen == 'zh':
                text = text.replace(' ', '')

        text = self.pre_caption(text, self.max_words, self.minimum_words)
        tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

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
