import copy
import os
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss

from einops import rearrange

from models.xvlm import XVLMBase, XVLMPlusBase
from models.xvlm import build_mlp


class XVLMForClassification(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False)

        feature_dim = self.vision_width if config.get('task_name') == 'imagenet' else self.text_width
        self.cls_head = build_mlp(input_dim=feature_dim, output_dim=config['num_labels'])

    def forward(self, image, text_ids, text_atts, targets=None, train=True):
        if image is None:
            output_cls = self.get_text_embeds_12L(text_ids, text_atts)[:, 0, :]

        elif text_ids is None:
            image_embeds, _ = self.get_vision_embeds(image)
            output_cls = image_embeds[:, 0, :]

        else:
            image_embeds, image_atts = self.get_vision_embeds(image)

            output_cls = self.get_cross_embeds(image_embeds, image_atts,
                                               text_ids=text_ids, text_atts=text_atts)[:, 0, :]

        prediction = self.cls_head(output_cls)
        if prediction.shape[-1] == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            return loss_fct(prediction.view(-1), targets.view(-1)) if train else prediction

        return F.cross_entropy(prediction, targets, ignore_index=-100) if train else prediction


class XVLMForVQAClassification(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False)

        self.cls_head = build_mlp(input_dim=self.text_width, output_dim=config['num_labels'])
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def forward(self, image, text_ids, text_atts, targets=None, k=None, weights=None, train=True, answer_pred=None,
                return_logits=False):

        image_embeds, image_atts = self.get_vision_embeds(image)
        output_cls = self.get_cross_embeds(image_embeds, image_atts,
                                           text_ids=text_ids, text_atts=text_atts)[:, 0, :]

        prediction = self.cls_head(output_cls)
        if train:
            if answer_pred is not None:
                self.criterion = nn.KLDivLoss(reduction='none')
                log_probs = F.log_softmax(prediction, -1)
                answer_label = F.softmax(answer_pred, dim=-1)
                loss = self.criterion(log_probs, answer_label)
                loss = loss.sum() / image.size(0)
                return loss

            p_states = []
            for b, n in enumerate(k):
                p_states = p_states + [prediction[b]] * n

            p_states = torch.stack(p_states, 0)

            loss = F.cross_entropy(p_states, targets, ignore_index=-100, reduction='none')

            loss = weights * loss
            loss = loss.sum() / image.size(0)
            if return_logits:
                return loss, prediction
            return loss
        else:
            return prediction


class XVLMForNLVR(XVLMBase):
    """
    Follow VLMo
    """
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=None)

        self.cls_head = build_mlp(input_dim=self.text_width * 2, output_dim=2)
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def forward(self, image, text_ids, text_atts, targets, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)
        image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

        output_cls_image1 = self.get_cross_embeds(image0_embeds, image_atts[:image0_embeds.size(0)],
                                                  text_ids=text_ids, text_atts=text_atts)[:, 0, :]

        output_cls_image2 = self.get_cross_embeds(image1_embeds, image_atts[image0_embeds.size(0):],
                                                  text_ids=text_ids, text_atts=text_atts)[:, 0, :]

        output_cls = torch.cat((output_cls_image1, output_cls_image2), dim=-1)

        assert output_cls.shape[-1] == self.text_width * 2

        prediction = self.cls_head(output_cls)

        return F.cross_entropy(prediction, targets) if train else prediction


class XVLMPlus4XVNLI(XVLMPlusBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False)

        self.cls_head = build_mlp(input_dim=self.text_width, output_dim=config['num_labels'])
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def forward(self, image, text_ids, text_atts, targets, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_cross_embeds(image_embeds, image_atts, text_ids, text_atts=text_atts)
        prediction = self.cls_head(text_embeds[:, 0, :])

        return F.cross_entropy(prediction, targets) if train else prediction


class XVLMPlusForMARVL(XVLMPlusBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False)

        self.cls_head = build_mlp(input_dim=self.text_width * 2, output_dim=2)
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def forward(self, image, text_ids, text_atts, targets, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)
        image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

        output_cls_image1 = self.get_cross_embeds(image0_embeds, image_atts[:image0_embeds.size(0)],
                                                  text_ids=text_ids, text_atts=text_atts)[:, 0, :]

        output_cls_image2 = self.get_cross_embeds(image1_embeds, image_atts[image0_embeds.size(0):],
                                                  text_ids=text_ids, text_atts=text_atts)[:, 0, :]

        output_cls = torch.cat((output_cls_image1, output_cls_image2), dim=-1)

        assert output_cls.shape[-1] == self.text_width * 2

        prediction = self.cls_head(output_cls)

        return F.cross_entropy(prediction, targets) if train else prediction

