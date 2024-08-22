# X^2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks (https://arxiv.org/abs/2211.12402)
# Github: https://github.com/zengyan-97/X2-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

# Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training (https://arxiv.org/abs/2206.00621)
# Github: https://github.com/zengyan-97/CCLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import json
import torch
from einops import rearrange

from models.xvlm import XVLMBase, XVLMPlusBase, VanillaConfig


class XVLM(XVLMBase):
    def __init__(self, config, load_vision_params=True, load_text_params=True, pretraining=True):
        super().__init__(config, load_vision_params=load_vision_params, load_text_params=load_text_params,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True,
                         config_text=None, pretraining=pretraining)

    def forward_multimodal(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None,
                           ret_bbox_loss=False, ret_match_loss=True):

        if ret_bbox_loss:
            image_embeds, image_atts, image_embeds_fullatts = \
                self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
        else:
            image_embeds, image_atts = self.get_vision_embeds(image)

        text_embeds = self.get_text_embeds(text_ids, text_atts)

        # with torch.no_grad():  # fix: i put it in batch iteration, so once a iteration
        #     self.temp.clamp_(0.001, 0.5)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat)

        if ret_match_loss:
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat)
        else:
            loss_itm = torch.tensor(0.0)

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)

        loss = {'loss_itc': loss_itc, 'loss_itm': loss_itm, 'loss_mlm': loss_mlm}

        if ret_bbox_loss:
            output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts)
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)

            loss['loss_bbox'] = loss_bbox
            loss['loss_giou'] = loss_giou

        return loss

    def forward_text(self, text_ids=None, text_atts=None,
                     text_ids_masked=None, masked_pos=None, masked_ids=None):

        loss = self.get_mlm_loss(text_ids_masked, text_atts, None, None, masked_pos, masked_ids)

        return {'loss_mlm': loss}

    def forward(self, image=None, text_ids=None, text_atts=None,
                text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None,
                ret_bbox_loss=False, ret_match_loss=True):

        if image is None:  # text
            loss = self.forward_text(text_ids, text_atts, text_ids_masked,
                                     masked_pos, masked_ids)

        else:
            loss = self.forward_multimodal(image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids,
                                           image_atts, idx_to_group_img, target_bbox, is_image, ret_bbox_loss,
                                           ret_match_loss=ret_match_loss)

        return loss


class XVLMPlus(XVLMPlusBase):
    def __init__(self, config, use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True,
                 load_vision_params=True, load_text_params=True, load_cross_params=True, pretraining=True):
        super().__init__(config, use_contrastive_loss=use_contrastive_loss, use_matching_loss=use_matching_loss,
                         use_mlm_loss=use_mlm_loss, use_bbox_loss=use_bbox_loss,
                         load_vision_params=load_vision_params, load_text_params=load_text_params, load_cross_params=load_cross_params,
                         pretraining=pretraining)

    def forward_multimodal(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None,
                           ret_bbox_loss=False, ret_match_loss=True):

        if ret_bbox_loss:
            image_embeds, image_atts, image_embeds_fullatts = \
                self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
        else:
            image_embeds, image_atts = self.get_vision_embeds(image)

        text_embeds = self.get_text_embeds(text_ids, text_atts)

        # with torch.no_grad():  # fix: i put it in batch iteration, so once a iteration
        #     self.temp.clamp_(0.001, 0.5)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat)

        if ret_match_loss:
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat)
        else:
            loss_itm = torch.tensor(0.0)

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)

        loss = {'loss_itc': loss_itc, 'loss_itm': loss_itm, 'loss_mlm': loss_mlm}

        if ret_bbox_loss:
            output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts)
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)

            loss['loss_bbox'] = loss_bbox
            loss['loss_giou'] = loss_giou

        return loss

    def forward(self, image=None, text_ids=None, text_atts=None,
                text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None,
                ret_bbox_loss=False, ret_match_loss=True):

        loss = self.forward_multimodal(image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids,
                                       image_atts, idx_to_group_img, target_bbox, is_image, ret_bbox_loss,
                                       ret_match_loss=ret_match_loss)

        return loss


class CrossViewLM(XVLMPlus):  # Multilingual x Multimodal Pre-training
    """
    Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training
    https://arxiv.org/abs/2206.00621
    """
    def __init__(self, config, use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True,
                 load_vision_params=True, load_text_params=True, load_cross_params=True, pretraining=True):
        super().__init__(config, use_contrastive_loss=use_contrastive_loss, use_matching_loss=use_matching_loss,
                         use_mlm_loss=use_mlm_loss, use_bbox_loss=use_bbox_loss,
                         load_vision_params=load_vision_params, load_text_params=load_text_params, load_cross_params=load_cross_params,
                         pretraining=pretraining)

    def forward_para_text(self, text_ids=None, text_atts=None,
                          text_ids_masked=None, text_atts_masked=None, masked_pos=None, masked_ids=None,
                          text_ids_2=None, text_atts_2=None, text_ids_masked_2=None, masked_pos_2=None, masked_ids_2=None):

        text_embeds = self.get_text_embeds(text_ids, text_atts)
        text_embeds_2 = self.get_text_embeds(text_ids_2, text_atts_2)

        # with torch.no_grad():
        #     self.temp.clamp_(0.001, 0.5)

        text_feat = self.get_features(text_embeds=text_embeds)
        text_feat_2 = self.get_features(text_embeds=text_embeds_2)

        loss_ttc = self.get_contrastive_loss(text_feat, text_feat_2)
        loss_ttm = self.get_matching_loss(text_embeds, text_atts, text_feat, text_embeds_2, text_atts_2, text_feat_2)

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, text_embeds_2, text_atts_2, masked_pos, masked_ids)

        loss = {'loss_ttc': loss_ttc, 'loss_ttm': loss_ttm, 'loss_mlm': loss_mlm}

        return loss

    def forward(self, image=None, text_ids=None, text_atts=None,
                text_ids_masked=None, text_atts_masked=None, masked_pos=None, masked_ids=None,
                text_ids_2=None, text_atts_2=None, text_ids_masked_2=None, masked_pos_2=None, masked_ids_2=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None, ret_bbox_loss=False, ret_match_loss=True):

        if image is None:  # parallel text
            loss = self.forward_para_text(text_ids, text_atts, text_ids_masked, text_atts_masked, masked_pos, masked_ids,
                          text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2)

        else:
            loss = self.forward_multimodal(image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids,
                                           image_atts, idx_to_group_img, target_bbox, is_image, ret_bbox_loss,
                                           ret_match_loss=ret_match_loss)

        return loss
