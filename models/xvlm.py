# X^2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks (https://arxiv.org/abs/2211.12402)
# Github: https://github.com/zengyan-97/X2-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn import CrossEntropyLoss

from einops import rearrange

from timm.models.layers import trunc_normal_

from models import box_ops

from models.xbert import BertConfig, BertForMaskedLM, BertModel
from models.xroberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
import copy

from utils import read_json
from dataset import build_tokenizer


class VanillaConfig(object):
    def __init__(self):
        pass


def load_params_change_prefix(state_dict: dict, prefix: str, new_prefix: str):
    if prefix == new_prefix:
        return state_dict

    state_dict_new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k.replace(prefix, new_prefix)

        state_dict_new[k] = v

    return state_dict_new


def load_roberta_lm_head(state_dict):
    def _replace(old_key: str, new_key: str):
        if new_key != old_key:
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]

    _replace('lm_head.bias', 'cls.predictions.bias')
    _replace('lm_head.dense.weight', 'cls.predictions.transform.dense.weight')
    _replace('lm_head.dense.bias', 'cls.predictions.transform.dense.bias')
    _replace('lm_head.layer_norm.weight', 'cls.predictions.transform.LayerNorm.weight')
    _replace('lm_head.layer_norm.bias', 'cls.predictions.transform.LayerNorm.bias')
    _replace('lm_head.decoder.weight', 'cls.predictions.decoder.weight')


def rename_tf_layernorm(state_dict):
    for k in list(state_dict.keys()):
        if 'LayerNorm.' in k:
            new_k = k.strip().replace('LayerNorm.beta', 'LayerNorm.bias')
            new_k = new_k.strip().replace('LayerNorm.gamma', 'LayerNorm.weight')
            state_dict[new_k] = state_dict[k]
            if new_k != k:
                del state_dict[k]


def load_params_choose_layers(prefix: str, state_dict: dict, mapper: dict, do_expand=False):
    """
        mapper: {old_layer: new_layer}
    """
    # fixed a bug
    # when mapper is for example {0: 0, 2: 1, 4: 2, 5: 3}
    # in the case, 4 -> 2 -> 1, causes error

    assert len(set(mapper.values())) == len(mapper), f"{set(mapper.values())} != {len(mapper)}"  # no overlap

    k_list = sorted([int(k) for k in mapper.keys()])
    mapper = {k: mapper[k] for k in k_list}

    if not len(mapper):
        return state_dict

    param_sorted = []

    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            i_layer = k[len(prefix)+1:]
            i_layer = int(i_layer.strip().split('.')[0])
            param_sorted.append((k, i_layer))
        else:
            param_sorted.append((k, -1))  # any is ok

    param_sorted = sorted(param_sorted, key=lambda p: p[1])
    param_sorted = [p[0] for p in param_sorted]

    for k in param_sorted:  # must start from lower layers
        if k.startswith(prefix):
            new_k = None
            for i in mapper.keys():
                if k.startswith(f'{prefix}.{i}.'):
                    new_k = k.replace(f'{prefix}.{i}.', f'{prefix}.{mapper[i]}.')
                    break

            if new_k:
                state_dict[new_k] = state_dict[k]

            if (new_k != k) and (not do_expand):
                del state_dict[k]

    return state_dict


def get_bert_config(encoder_rpath, num_hidden_layers=12, cross_start_at=12):
    """
    Args:
        cross_start_at: if it >= num_hidden_layers, no cross attn
    """
    if 'roberta' in encoder_rpath:
        config = RobertaConfig.from_json_file(os.path.join(encoder_rpath, 'config.json'))
    else:
        config = BertConfig.from_json_file(os.path.join(encoder_rpath, 'config.json'))

    # set configs
    config.num_hidden_layers = num_hidden_layers
    config.fusion_layer = cross_start_at
    config.embedding_dim = config.hidden_size

    return config


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


def build_vision_encoder(config, load_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2

    if config.get('use_clip_vit', False):  # good performance, but only base model available
        from models.clip_vit import CLIPVisionTransformer, interpolate_pos_embed

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        vision_encoder = CLIPVisionTransformer(image_size=config['image_res'], patch_size=vision_config['patch_size'],
                                               hidden_size=vision_config['vision_width'],
                                               hidden_act=vision_config['hidden_act'],
                                               num_attention_heads=vision_config['num_attention_heads'],
                                               attention_dropout=vision_config['attention_dropout'],
                                               intermediate_size=vision_config['intermediate_size'],
                                               num_hidden_layers=vision_config['num_hidden_layers'],
                                               local_attn_depth=vision_config['local_attn_depth'])

        if load_params:
            # download from https://huggingface.co/openai/clip-vit-base-patch16/tree/main
            state_dict_orig = torch.load(vision_config['ckpt'], map_location="cpu")
            state_dict = {}
            for k, v in state_dict_orig.items():
                if k.startswith('vision_model.'):
                    k = k[13:]
                    if k.startswith('embeddings.'):
                        k = k[11:]
                        k = k.replace('patch_embedding.weight', 'patch_embed.weight')
                        k = k.replace('position_embedding.weight', 'pos_embed.weight')

                    if k != 'position_ids':
                        state_dict[k] = v

            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed.weight'].unsqueeze(dim=0),
                                                       num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

            assert vision_config['num_hidden_layers'] in [6, 12], "param initialization not implemented"
            if vision_config['num_hidden_layers'] == 6:
                mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                load_params_choose_layers('encoder.layers', state_dict, mapper)

    elif config.get('use_swin', False):
        from models.swin_transformer import SwinTransformer

        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False, add_cls=config.get('swin_add_cls', True))

        if load_params:
            from models.swin_transformer import load_pretrained_swin
            state_dict = load_pretrained_swin(vision_encoder, vision_config['ckpt'])

    elif config.get('use_beit_v2', False):

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        if 'base' in config['vision_config']:
            from models.beit2 import beit_base_patch16 as beit_model
        elif 'large' in config['vision_config']:
            from models.beit2 import beit_large_patch16 as beit_model
        else:
            raise ValueError

        vision_encoder = beit_model(img_size=config['image_res'],
                                    drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
                                    use_mean_pooling=True,
                                    init_scale=0.001,
                                    use_rel_pos_bias=True, use_abs_pos_emb=False,
                                    init_values=0.1, qkv_bias=True, local_attn_depth=config.get('local_attn_depth', -1),
                                    vision_num_hidden_layers=config.get('vision_num_hidden_layers', -1))

        if load_params:
            from models.beit2 import load_pretrained_beit2
            load_pretrained_beit2(vision_encoder, vision_config['ckpt'])

    else:
        raise ValueError

    if load_params and (not config.get('use_beit_v2', False)):
        print("### Load ViT: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    # set attrs
    vision_encoder.vision_width = vision_width

    return vision_encoder


def build_text_encoder(config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
    if config_text is None:
        config_text = get_bert_config(config['text_encoder'], num_hidden_layers=config['text_num_hidden_layers'],
                                      cross_start_at=config['text_fusion_start_at'])
    else:
        assert isinstance(config_text, BertConfig)

    tokenizer = build_tokenizer(config['text_encoder'])
    config_text.pad_token_id = tokenizer.pad_token_id

    config_text.text_encoder = config['text_encoder']
    config_text.hidden_dropout_prob = config.get('dropout', config_text.hidden_dropout_prob)  # changable dropout rate
    config_text.encoder_width = vision_width
    config_text.text_drop_path_rate = config.get('text_drop_path_rate', 0.0)
    config_text.cross_drop_path_rate = config.get('cross_drop_path_rate', 0.0)

    if use_mlm_loss:
        # assert load_text_params is True  # for domain pre-training
        if ('accelerator' in config.keys()) and (config['accelerator']['FP16_OPT_LEVEL'] != 'O0'):
            config_text.fp16 = True  # will use some operations to avoid gradient overflow

        if 'roberta' in config['text_encoder']:
            text_encoder = RobertaForMaskedLM(config=config_text)
        else:
            text_encoder = BertForMaskedLM(config=config_text)

    else:
        if 'roberta' in config['text_encoder']:
            text_encoder = RobertaModel(config=config_text, add_pooling_layer=False)
        else:
            text_encoder = BertModel(config=config_text, add_pooling_layer=False)

    missing_keys = []
    if load_text_params:
        print("### Initializing text encoder from ", os.path.join(config['text_encoder'], 'pytorch_model.bin'))
        state_dict = torch.load(os.path.join(config['text_encoder'], 'pytorch_model.bin'), map_location="cpu")
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']

        prefix = "roberta.encoder.layer" if 'roberta' in config['text_encoder'] else 'bert.encoder.layer'
        if not use_mlm_loss:
            state_dict = {k.replace('roberta.', '').replace('bert.', ''): v for k, v in state_dict.items()}
            prefix = "encoder.layer"

        if 'roberta' in config['text_encoder']:
            pass
        else:
            if 'bert-base-uncased' in config['text_encoder']:
                rename_tf_layernorm(state_dict)
                if config_text.num_hidden_layers == 18:
                    assert config['text_fusion_start_at'] == 12
                    mapper = {6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 17}
                    load_params_choose_layers(prefix, state_dict, mapper, do_expand=True)

                else:
                    pass

            elif 'bert-large-uncased-12l' in config['text_encoder']:
                if config['text_num_hidden_layers'] == 18:
                    assert config['text_fusion_start_at'] == 12
                    mapper = {6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 17}
                    load_params_choose_layers(prefix, state_dict, mapper, do_expand=True)
                else:
                    raise NotImplementedError

            elif 'bert-large-uncased' in config['text_encoder']:
                rename_tf_layernorm(state_dict)

                if config_text.num_hidden_layers == 12:
                    mapper = {layer: i for i, layer in enumerate(list(range(1, 24+1, 2)))}
                    load_params_choose_layers(prefix, state_dict, mapper)

                else:
                    raise NotImplementedError

            elif 'chinese-roberta-wwm-ext' in config['text_encoder']:
                if config_text.num_hidden_layers == 6:
                    mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                    load_params_choose_layers(prefix, state_dict, mapper)

            else:
                raise NotImplementedError

        if config.get('init_word_embeddings', False):
            print("### Train word_embeddings from scratch...", flush=True)
            for k in list(state_dict.keys()):
                if 'word_embeddings' in k:
                    del state_dict[k]

                elif k == 'cls.predictions.decoder.weight':
                    del state_dict[k]

                elif k == 'cls.predictions.bias':
                    del state_dict[k]

        msg = text_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys, flush=True)
        print("unexpected_keys: ", msg.unexpected_keys, flush=True)

        missing_keys = msg.missing_keys

    return text_encoder, missing_keys


def load_pretrained(model, ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    if is_eval:
        return state_dict

    print("### Loading pretrained vision encoder", flush=True)

    if config.get('use_clip_vit', False):
        from models.clip_vit import interpolate_pos_embed
        del state_dict['vision_encoder.position_ids']
        num_patches = (config['image_res'] // config['patch_size']) ** 2
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config.get('use_swin', False) or config.get('use_swin_v2', False):
        from models.swin_transformer import load_pretrained_swin

        vision_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith('vision_encoder.'):
                vision_state_dict[k[15:]] = state_dict[k]
                del state_dict[k]

        vision_state_dict = load_pretrained_swin(model.vision_encoder, state_dict=vision_state_dict)

        for k in vision_state_dict.keys():
            state_dict['vision_encoder.' + k] = vision_state_dict[k]

    elif config.get('use_beit_v2', False):
        from models.beit2 import interpolate_pos_embed

        vision_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith('vision_encoder.'):
                vision_state_dict[k[15:]] = state_dict[k]
                del state_dict[k]

        vision_state_dict = interpolate_pos_embed(model.vision_encoder, vision_state_dict)
        for k in vision_state_dict.keys():
            state_dict['vision_encoder.' + k] = vision_state_dict[k]

    else:
        raise ValueError

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if key.startswith('text_encoder.') or key.startswith('cross_encoder.'):
                encoder_key = key.replace('roberta.', '').replace('bert.', '').strip()
                state_dict[encoder_key] = state_dict[key]
                if encoder_key != key:
                    del state_dict[key]

    if config.get('init_timesformer', False):
        map_dict = {
            "temporal_norm1": "norm1",
            "time_attn": "attn",
            "temporal_norm2": "norm2",
            "temporal_mlp": "mlp",
            "time_gamma_1": "gamma_1",
            "time_gamma_2": "gamma_2"
        }
        for from_key, to_key in map_dict.items():
            for key in list(state_dict.keys()):
                if to_key in key:
                    state_dict[key.replace(to_key, from_key)] = copy.deepcopy(state_dict[key])

    return state_dict


class XVLMBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=False, load_cross_params=False,
                 use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                 config_text=None, pretraining=False):
        super().__init__()

        self.init_params = []
        self.vision_encoder = self.build_vision_encoder(config, load_params=load_vision_params)
        self.vision_width = self.vision_encoder.vision_width

        self.text_encoder, missing_keys = self.build_text_encoder(config, vision_width=self.vision_width,
                                                                   load_text_params=load_text_params,
                                                                   use_mlm_loss=use_mlm_loss,
                                                                   config_text=config_text)
        self.update_init_params([f'text_encoder.{k}' for k in missing_keys])

        self.cross_encoder, missing_keys = self.build_cross_encoder(config, self.text_encoder.config, load_cross_params=load_cross_params)
        self.update_init_params([f'cross_encoder.{k}' for k in missing_keys])

        # Build Video Encoding
        self.video_encoding = config.get('video_encoding', '')
        if self.video_encoding == 'avgpool':
            self.video_pooling = nn.AdaptiveAvgPool1d(1)
        elif self.video_encoding == 'timesformer':
            self.video_pooling = nn.AdaptiveAvgPool1d(1)
        elif self.video_encoding == 'tubevit':
            self.video_pooling = nn.AdaptiveAvgPool1d(1)
        elif self.video_encoding == '':
            pass
        else:
            raise ValueError(f"Not Supported video_encoding == {config['video_encoding']}")

        if self.video_encoding != '':
            self.frame_len = config['frame_len']
            self.add_frame_pos = config['add_frame_pos']
            if self.add_frame_pos:
                self.absolute_frame_pos_embed = nn.Parameter(torch.zeros(1, self.frame_len, 1, self.vision_width))
                trunc_normal_(self.absolute_frame_pos_embed, std=.02)
                self.update_init_params(['absolute_frame_pos_embed'])

        self.use_contrastive_loss = use_contrastive_loss
        if self.use_contrastive_loss:
            self.embed_dim = config['embed_dim']
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
            self.update_init_params(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
            self.update_init_params(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            if config.get('fix_temp', False):
                self.temp = torch.ones([]) * config['temp']

            else:
                self.temp = nn.Parameter(torch.ones([]) * config['temp'])
                self.update_init_params(['temp'])

        self.use_matching_loss = use_matching_loss
        if self.use_matching_loss:
            self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)
            self.update_init_params(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

        self.use_bbox_loss = use_bbox_loss
        if self.use_bbox_loss:
            self.bbox_head = build_mlp(input_dim=self.text_width, output_dim=4)
            self.update_init_params(['bbox_head.' + n for n, _ in self.bbox_head.named_parameters()])

        if pretraining:
            print("Train From Scratch: ", sorted(self.init_params))
        else:
            self.init_params = []

    def build_vision_encoder(self, config, load_params=False):
        return build_vision_encoder(config, load_params=load_params)

    def build_text_encoder(self, config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
        """
        in XVLMBase, text_encoder includes cross encoder parts
        """
        text_encoder, missing_keys = build_text_encoder(config, vision_width, load_text_params=load_text_params,
                                  use_mlm_loss=use_mlm_loss, config_text=config_text)

        # set attrs
        self.vocab_size = text_encoder.config.vocab_size
        self.num_text_layers = text_encoder.config.fusion_layer
        self.text_width = text_encoder.config.hidden_size  # i.e. cross_width
        print("### X-VLM, num_text_layers: ", self.num_text_layers, flush=True)

        return text_encoder, missing_keys

    def build_cross_encoder(self, config, config_text, load_cross_params=False):
        """
        in XVLMBase, text_encoder includes cross encoder parts
        """

        cross_encoder = None
        missing_keys = []

        # set attrs
        self.num_cross_layers = self.text_encoder.config.num_hidden_layers - self.num_text_layers
        self.cross_width = self.text_encoder.config.hidden_size
        print("### X-VLM, num_cross_layers: ", self.num_cross_layers, flush=True)

        return cross_encoder, missing_keys

    def update_init_params(self, missing_keys=None):
        if missing_keys is not None:
            assert isinstance(missing_keys, list)
            for k in missing_keys:
                if k not in self.init_params:
                    self.init_params.append(k)

        # check
        named_parameters = set([n for n, _ in self.named_parameters()])
        for n in set(self.init_params):
            if n not in named_parameters:
                self.init_params.remove(n)

    def load_pretrained(self, ckpt_rpath, config, is_eval=False, is_domain_pretrain=False):
        print('load checkpoint from %s' % ckpt_rpath)
        if is_domain_pretrain:
            checkpoint = torch.load(ckpt_rpath, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        
            if config.get('init_timesformer', False):
                map_dict = {
                    "temporal_norm1": "norm1",
                    "time_attn": "attn",
                    "temporal_norm2": "norm2",
                    "temporal_mlp": "mlp",
                    "time_gamma_1": "gamma_1",
                    "time_gamma_2": "gamma_2"
                }
                for from_key, to_key in map_dict.items():
                    for key in list(state_dict.keys()):
                        if to_key in key:
                            state_dict[key.replace(to_key, from_key)] = copy.deepcopy(state_dict[key])
        else:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=is_eval, load_text=True)

        if hasattr(self, 'absolute_frame_pos_embed') and ('absolute_frame_pos_embed' in state_dict.keys()):
            pretrained = state_dict['absolute_frame_pos_embed']
            if pretrained.shape != self.absolute_frame_pos_embed.shape:
                frame_len = min(pretrained.shape[1], self.absolute_frame_pos_embed.shape[1])
                self.absolute_frame_pos_embed.data[:, :frame_len, :, :] = pretrained.data[:, :frame_len, :, :]
                print(f"load absolute_frame_pos_embed[:{frame_len}] ({pretrained.shape[1]}/{self.absolute_frame_pos_embed.shape[1]})", flush=True)
                del state_dict['absolute_frame_pos_embed']

        msg = self.load_state_dict(state_dict, strict=False)
        print("unexpected_keys: ", msg.unexpected_keys)
        missing_keys = [p for p in msg.missing_keys]  # if 'vision_encoder' not in p
        self.update_init_params(missing_keys)
        print("train from scratch: ", sorted(self.init_params))

    def _encode_frames(self, frames, output_hidden_states=None, output_attentions=None):
        assert frames.dim() == 5  # (bsz, frame_len, c, h, w)

        bsz = frames.shape[0]

        frames = rearrange(frames, 'b f c h w -> (b f) c h w')

        outputs = self.vision_encoder(frames, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        frame_embeds = outputs['last_hidden_state'] if output_hidden_states else outputs
        frame_embeds = rearrange(frame_embeds, '(b f) p d -> b f p d', b=bsz)

        if self.add_frame_pos:
            frame_embeds = frame_embeds + self.absolute_frame_pos_embed

        if output_hidden_states:
            return frame_embeds, outputs

        return frame_embeds, None  # bsz, frame_len, patch_len, d

    def _encode_video_pooling(self, frames, output_hidden_states=None, output_attentions=None):
        bsz = frames.shape[0]
        frame_embeds, outputs = self._encode_frames(frames, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        frame_embeds = rearrange(frame_embeds, 'b f p d -> (b p) d f')

        # frame_embeds: bsz * patch_len, d, frame_len
        frame_embeds = self.video_pooling(frame_embeds)
        # frame_embeds: bsz * patch_len, d, 1

        return rearrange(frame_embeds, '(b p) d 1 -> b p d', b=bsz), outputs

    def get_frame_embeds(self, frame, output_hidden_states=None, output_attentions=None):
        assert frame.dim() == 5
        assert output_hidden_states == output_attentions

        if self.video_encoding == 'avgpool':
            frame_embeds, outputs = self._encode_video_pooling(frame, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        else:
            raise NotImplementedError(f"video_encoding == '{self.video_encoding}'")

        frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(frame_embeds.device)

        if output_hidden_states:
            return frame_embeds, frame_atts, outputs

        return frame_embeds, frame_atts

    def get_image_embeds(self, image, image_atts=None, idx_to_group_img=None, output_hidden_states=None, output_attentions=None):
        assert image.dim() == 4
        assert output_hidden_states == output_attentions

        if idx_to_group_img is None:
            image_embeds = self.vision_encoder(image, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            return image_embeds, image_atts  # full attention

        else:  # image < bsz
            if output_attentions or output_hidden_states:
                raise NotImplementedError

            if image_atts is None:
                image_embeds_fullatts = self.vision_encoder(image)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))  # expend to bsz

                image_atts = torch.ones(image_embeds_fullatts.size()[:-1], dtype=torch.long).to(image.device)

                return image_embeds_fullatts, image_atts

            else:
                assert image_atts.size(0) == idx_to_group_img.size(0)  # bsz
                image_embeds, image_embeds_fullatts = \
                    self.vision_encoder(image, idx_to_group_img=idx_to_group_img, image_atts=image_atts)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                return image_embeds, image_atts, image_embeds_fullatts

    def get_vision_embeds(self, image, image_atts=None, idx_to_group_img=None, output_hidden_states=None, output_attentions=None):
        """
        vision_embeds: cls + patch embeds
        """
        assert output_hidden_states == output_attentions

        if image.dim() == 5:  # encode video
            # image: (bsz, frame_len, c, h, w)
            assert idx_to_group_img is None, "not supported"
            return self.get_frame_embeds(image, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        assert image.dim() == 4
        return self.get_image_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img,
                                     output_hidden_states=output_hidden_states, output_attentions=output_attentions)

    def get_text_embeds(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        outputs = encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text',
                          output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        else:
            return outputs.last_hidden_state

    def get_text_embeds_12L(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        outputs = encoder(text_ids,
                       attention_mask=text_atts,
                       encoder_hidden_states=None,
                       encoder_attention_mask=None,
                       output_hidden_states=output_hidden_states,
                       output_attentions=output_attentions,
                       return_dict=True)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        else:
            return outputs.last_hidden_state

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None, output_hidden_states=None, output_attentions=None):
        assert text_atts is not None
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        if text_embeds is not None:
            outputs = encoder(encoder_embeds=text_embeds,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           mode='fusion')

        elif text_ids is not None:
            outputs = encoder(text_ids,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True)

        else:
            raise ValueError

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        return outputs.last_hidden_state

    def get_features(self, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        else:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1), \
                   F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_hard_negatives(self, image_feat, text_feat, idx=None):
        bs = image_feat.size(0)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_neg_idx = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_neg_idx.append(neg_idx)

        text_neg_idx = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_neg_idx.append(neg_idx)

        return image_neg_idx, text_neg_idx

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        """
        Matching Loss with hard negatives
        """
        image_neg_idx, text_neg_idx = self.get_hard_negatives(image_feat, text_feat, idx=idx)

        bs = image_feat.size(0)
        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = image_neg_idx[b]
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = text_neg_idx[b]
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts)[:, 0, :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,
                                          text_atts=text_atts_all)[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        return F.cross_entropy(output, itm_labels)

    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss

    def predict_bbox(self, image_embeds, text_embeds, text_atts):
        """
        Args:
            image_embeds: encoding full images

        Returns:
            output_coord: bsz, 4
        """
        assert image_embeds.size(0) == text_embeds.size(0)

        output_cls = self.get_cross_embeds(image_embeds, torch.ones(image_embeds.shape[:2]).to(image_embeds.device),
                                           text_embeds=text_embeds, text_atts=text_atts)[:, 0, :]

        output_coord = self.bbox_head(output_cls).sigmoid()

        return output_coord

    def get_bbox_loss(self, output_coord, target_bbox_ex, is_image=None, target_bbox_map_ids=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        n_objs = output_coord.size(0)
        n_bbox = target_bbox_ex.size(0)

        assert n_objs == n_bbox
        target_bbox = target_bbox_ex

        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes


class XVLMPlusBase(XVLMBase):
    """
    Separate text encoder and cross encoder, making the text encoder easier to be replaced.
    Re-implement func build_text_encoder to support any type of text encoder
    """
    def __init__(self, config, load_vision_params=False, load_text_params=False, load_cross_params=False,
                 use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                 pretraining=False):

        super().__init__(config, load_vision_params=load_vision_params, load_text_params=load_text_params, load_cross_params=load_cross_params,
                 use_contrastive_loss=use_contrastive_loss, use_matching_loss=use_matching_loss, use_mlm_loss=False, use_bbox_loss=use_bbox_loss,
                 config_text=None, pretraining=pretraining)

        self.use_mlm_loss = use_mlm_loss
        if use_mlm_loss:
            from models.xbert import BertOnlyMLMHead
            self.mlm_head = BertOnlyMLMHead(self.text_encoder.config)
            self.update_init_params(['mlm_head.' + n for (n, _) in self.mlm_head.named_parameters()])
            # self.tie_text_and_cross_wordemb()

        if pretraining:
            print("### Train From Scratch: ", sorted(self.init_params))
        else:
            self.init_params = []

    def build_text_encoder(self, config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
        config_text = get_bert_config(config['text_encoder'], num_hidden_layers=config['text_num_hidden_layers'],
                                      cross_start_at=config['text_num_hidden_layers'])

        text_encoder, missing_keys = build_text_encoder(config, vision_width, load_text_params=load_text_params,
                                  use_mlm_loss=use_mlm_loss, config_text=config_text)

        # set attrs
        self.vocab_size = text_encoder.config.vocab_size
        self.num_text_layers = text_encoder.config.fusion_layer
        self.text_width = text_encoder.config.hidden_size  # i.e. cross_width
        print("### X-VLM, num_text_layers: ", self.num_text_layers, flush=True)

        return text_encoder, missing_keys

    def build_cross_encoder(self, config, config_text, load_cross_params=False):
        config_cross = get_bert_config(config['cross_encoder'], num_hidden_layers=config['cross_num_hidden_layers'],
                                       cross_start_at=0)

        if config_text.hidden_size != config_cross.hidden_size:
            raise ValueError

        config_cross.pad_token_id = config_text.pad_token_id
        config_cross.vocab_size = config_text.vocab_size
        config_cross.embedding_dim = config_text.embedding_dim
        config_cross.encoder_width = config_text.encoder_width

        assert 'cross_drop_path_rate' not in config, "notimplemented"

        cross_encoder = BertModel(config=config_cross, add_pooling_layer=False, add_embeddings_layer=False)

        self.num_cross_layers = cross_encoder.config.num_hidden_layers
        self.cross_width = cross_encoder.config.hidden_size
        print("### X-VLM, num_cross_layers: ", self.num_cross_layers, flush=True)

        missing_keys = []
        if load_cross_params:
            print("### Initializing cross encoder from ", os.path.join(config['cross_encoder'], 'pytorch_model.bin'))
            state_dict = torch.load(os.path.join(config['cross_encoder'], 'pytorch_model.bin'), map_location="cpu")
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']

            state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
            prefix = "encoder.layer"

            if 'bert-base-uncased' in config['cross_encoder']:
                rename_tf_layernorm(state_dict)
                if config_cross.num_hidden_layers == 6:
                    mapper = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}
                    load_params_choose_layers(prefix, state_dict, mapper)
                else:
                    raise NotImplementedError

            elif 'bert-large-uncased-12l' in config['cross_encoder']:
                if config_cross.num_hidden_layers == 6:
                    mapper = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}
                    load_params_choose_layers(prefix, state_dict, mapper)
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            # del
            for k in list(state_dict.keys()):
                if 'word_embeddings' in k:
                    del state_dict[k]
                elif k == 'cls.predictions.decoder.weight':
                    del state_dict[k]
                elif k == 'cls.predictions.bias':
                    del state_dict[k]

            msg = cross_encoder.load_state_dict(state_dict, strict=False)
            print("missing_keys: ", msg.missing_keys, flush=True)
            print("unexpected_keys: ", msg.unexpected_keys, flush=True)

            missing_keys = [f'cross_encoder.{k}' for k in msg.missing_keys]

        return cross_encoder, missing_keys

    def tie_text_and_cross_wordemb(self):
        # # tie word embeddings
        # self.mlm_head.predictions.decoder.weight = self.text_encoder.embeddings.word_embeddings.weight
        # self.mlm_head.predictions.decoder.bias = self.text_encoder.embeddings.word_embeddings.bias
        # self.init_params.remove('mlm_head.predictions.decoder.weight')
        # self.init_params.remove('mlm_head.predictions.decoder.bias')
        raise NotImplementedError("implement it in the domain pretraining model")

    def load_pretrained_xvlm(self, xvlm_ckpt_rpath, config, is_eval=False):
        print('### Loading X-VLM checkpoint from %s' % xvlm_ckpt_rpath, flush=True)
        state_dict = load_pretrained(self, xvlm_ckpt_rpath, config, is_eval=is_eval, load_text=True)

        if hasattr(self, 'absolute_frame_pos_embed') and ('absolute_frame_pos_embed' in state_dict.keys()):
            pretrained = state_dict['absolute_frame_pos_embed']
            if pretrained.shape != self.absolute_frame_pos_embed.shape:
                frame_len = min(pretrained.shape[1], self.absolute_frame_pos_embed.shape[1])
                self.absolute_frame_pos_embed.data[:, :frame_len, :, :] = pretrained.data[:, :frame_len, :, :]
                print(
                    f"load absolute_frame_pos_embed[:{frame_len}] ({pretrained.shape[1]}/{self.absolute_frame_pos_embed.shape[1]})",
                    flush=True)
                del state_dict['absolute_frame_pos_embed']

        replace_text_encoder = config.get('replace_text_encoder', False)
        num_text_layers = config['xvlm_ckpt_text_num_hidden_layers']
        if not is_eval:
            for k in list(state_dict.keys()):
                if k.startswith('text_encoder.'):
                    if 'layer' in k:
                        encoder_keys = k.split('.')
                        layer_num = int(encoder_keys[3])
                        if layer_num < num_text_layers:
                            if replace_text_encoder:
                                del state_dict[k]  # del text encoder
                        else:
                            new_k = k.replace('text_encoder.', 'cross_encoder.')
                            new_k = new_k.replace(f'layer.{layer_num}.', f'layer.{layer_num-num_text_layers}.')
                            state_dict[new_k] = state_dict[k]
                            del state_dict[k]
                    else:

                        if replace_text_encoder:
                            if 'embeddings' in k:
                                del state_dict[k]
                                continue
                            elif k.startswith('text_encoder.cls.predictions.decoder'):
                                del state_dict[k]
                                continue
                            elif k.startswith('text_encoder.cls.predictions.bias'):
                                del state_dict[k]
                                continue

                        if k.startswith('text_encoder.cls.'):
                            new_k = k.replace('text_encoder.cls.', 'mlm_head.').strip()
                            state_dict[new_k] = state_dict[k]
                            del state_dict[k]

        return state_dict

    def load_pretrained_text(self, text_ckpt_rpath, config, is_eval=False):
        assert os.path.exists(text_ckpt_rpath)
        print('### Loading Text-Encoder checkpoint from %s' % text_ckpt_rpath)
        checkpoint = torch.load(text_ckpt_rpath, map_location='cpu')
        text_enc_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        rename_tf_layernorm(text_enc_state_dict)

        for k in list(text_enc_state_dict.keys()):
            if k.startswith('roberta.') or k.startswith('bert.'):
                encoder_key = 'text_encoder.' + k.replace('roberta.', '').replace('bert.', '').strip()

                text_enc_state_dict[encoder_key] = text_enc_state_dict[k]
                if encoder_key != k:
                    del text_enc_state_dict[k]

        return text_enc_state_dict

    def load_pretrained(self, xvlm_ckpt_rpath, config, is_eval=False, text_ckpt_rpath=''):
        """
        xvlm_ckpt_rpath: of XVLMBase
        """
        if config.get('is_xvlm_ckpt', False):
            state_dict = self.load_pretrained_xvlm(xvlm_ckpt_rpath, config, is_eval=is_eval)
        else:
            state_dict = load_pretrained(self, xvlm_ckpt_rpath, config, is_eval=is_eval, load_text=True)

        if config.get('replace_text_encoder', False):
            text_enc_state_dict = self.load_pretrained_text(text_ckpt_rpath, config, is_eval=is_eval)
            for k, v in text_enc_state_dict.items():
                state_dict[k] = v

        msg = self.load_state_dict(state_dict, strict=False)
        print("unexpected_keys: ", msg.unexpected_keys)
        missing_keys = [p for p in msg.missing_keys]  #  if 'vision_encoder' not in p
        self.update_init_params(missing_keys)
        print("train from scratch: ", sorted(self.init_params))

    def get_text_embeds(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        assert output_hidden_states == output_attentions

        outputs = self.text_encoder(text_ids, attention_mask=text_atts, return_dict=True,
                                    output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        return outputs.last_hidden_state

    def get_text_embeds_12L(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        return self.get_text_embeds(text_ids, text_atts, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None,
                         output_hidden_states=None, output_attentions=None):
        assert text_atts is not None
        assert output_hidden_states == output_attentions

        if text_embeds is None:
            assert text_ids is not None
            assert not output_hidden_states, "please manually split get_text_embeds and get_cross_embeds"
            text_embeds = self.get_text_embeds(text_ids, text_atts)

        outputs = self.cross_encoder(encoder_embeds=text_embeds,
                       attention_mask=text_atts,
                       encoder_hidden_states=image_embeds,
                       encoder_attention_mask=image_atts,
                       return_dict=True,
                       mode='fusion',
                       output_hidden_states=output_hidden_states,
                       output_attentions=output_attentions)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        return outputs.last_hidden_state

    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        sequence_output = self.get_cross_embeds(image_embeds, image_atts, text_ids=text_ids_masked, text_atts=text_atts)

        # sequence_output, (bs, len, 768)
        # masked_pos, (bs, n_mask)
        sequence_output = gather_seq_out_by_pos(sequence_output, masked_pos)
        # sequence_output, (bs, n_mask, 768)

        prediction_scores = self.mlm_head(sequence_output)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_ids.view(-1))

        return masked_lm_loss

