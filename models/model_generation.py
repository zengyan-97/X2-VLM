import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from dataset import build_tokenizer
from models.xbert import BertConfig, BertLMHeadModel

from models.xvlm import XVLMBase, XVLMPlusBase, load_pretrained


class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.float().repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)


class XVLMForMLMCaptioning(XVLMBase):
    """
    MLM generation based on images

    Following
        An Investigation of Suitability of Pre-Trained Language Models for Dialogue Generation–Avoiding Discrepancies
        https://aclanthology.org/2021.findings-acl.393.pdf

    """
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=True, use_bbox_loss=False, config_text=None)

        self.tokenizer = build_tokenizer(config['text_encoder'])

        self.prompt_ids = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token] + self.tokenizer.tokenize(config['prompt']))

        print("### prompt_ids, ", self.prompt_ids, flush=True)

        # cls_token will not be masked in dataset preprocessing
        self.crit_mask_lm_smoothed = LabelSmoothingLoss(
            config['label_smoothing'], self.tokenizer.vocab_size, ignore_index=self.tokenizer.cls_token_id, reduction='none')

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(self, ckpt_rpath, config, load_text=False)

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, input_ids_masked, attention_mask, position_ids, masked_pos, masked_ids, masked_weight):
        image_embeds, image_atts = self.get_vision_embeds(image)

        prediction_scores_masked = self.text_encoder(
            input_ids_masked, attention_mask=attention_mask, position_ids=position_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts, masked_pos=masked_pos, return_logits=True)
        # prediction_scores_masked: bsz, n_mask, vocab_size

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        masked_lm_loss = self.crit_mask_lm_smoothed(
            F.log_softmax(prediction_scores_masked, dim=-1), masked_ids)

        pseudo_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weight)

        return pseudo_lm_loss

    def generate(self, image, num_beams=3, min_length=5, max_length=20,
                 length_penalty=0, forbid_duplicate_ngrams=True, ngram_size=3):

        bsz = image.size(0)
        input_ids = torch.tensor(self.prompt_ids, dtype=torch.long, device=image.device).view(1, -1).expand(bsz, -1)

        length = input_ids.size(0)+max_length
        token_type_ids = torch.zeros((bsz, length), dtype=torch.long, device=image.device)
        position_ids = torch.tensor(range(length), dtype=torch.long, device=image.device).view(1, -1).expand(bsz, -1)
        attention_mask = torch.tril(torch.ones((length, length), dtype=torch.long,
                                               device=image.device)).view(1, length, length).expand(bsz, length, length)

        output_ids = self.beam_search(image, input_ids, token_type_ids, position_ids, attention_mask,
                                      num_beams=num_beams, min_length=min_length, length_penalty=length_penalty,
                                      forbid_duplicate_ngrams=forbid_duplicate_ngrams, ngram_size=ngram_size)

        def _get_captions(caption_ids):
            captions = []
            for output in caption_ids:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                # caption = caption[len(self.prompt):]
                captions.append(caption)
            return captions

        return _get_captions(output_ids)

    def beam_search(self, image, input_ids, token_type_ids, position_ids, attention_mask,
                    num_beams=3, min_length=5, length_penalty=0, forbid_duplicate_ngrams=True, ngram_size=3,
                    ):
        """
        modified from https://github.com/microsoft/unilm/tree/master/s2s-ft
        """
        image_embeds, image_atts = self.get_vision_embeds(image)

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.tokenizer.mask_token_id)
        next_pos = input_length

        K = num_beams

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []

        forbid_word_mask = None
        buf_matrix = None
        forbid_ignore_set = set()

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]

            outputs = self.text_encoder.bert(x_input_ids,
                                             attention_mask=curr_attention_mask,
                                             token_type_ids=curr_token_type_ids,
                                             position_ids=curr_position_ids,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             output_hidden_states=True,
                                             history_states=prev_encoded_layers,
                                             is_decoder=True, return_dict=True)

            new_encoded_layers = outputs.hidden_states

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.text_encoder.cls(last_hidden)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)

            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if min_length and (next_pos - input_length + 1 <= min_length):
                log_scores[:, :, self.tokenizer.eos_token_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)

                # back_ptrs = torch.div(k_ids, K)  # fixed->
                back_ptrs = torch.div(k_ids, K, rounding_mode='floor')

                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.tokenizer.eos_token_id).type_as(kk_scores))
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])

                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_encoded_layers is None)

            if is_first:
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)

                image_embeds = first_expand(image_embeds)
                image_atts = first_expand(image_atts)

            if forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n - 1):]
                    if forbid_ignore_set and any(tk in forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not (
                                forbid_ignore_set and (seq[i + n - 1] in forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.tokenizer.eos_token_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.tokenizer.eos_token_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0, length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        traces = {k: v.tolist() for k, v in traces.items()}
        output_ids = traces['pred_seq']
        return output_ids


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))


class XVLMForVQA(XVLMBase):
    """
    Generative Model, but model the task as ranking at inference
    Use XVLMForGeneration for purely generative tasks.
    """
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=None)

        assert isinstance(config['pad_token_id'], int)
        self.pad_token_id = config['pad_token_id']
        config_enc = self.text_encoder.config

        self.num_text_layers = config_enc.fusion_layer
        self.num_cross_layers = config_enc.num_hidden_layers - config_enc.fusion_layer
        assert config['num_dec_layers'] in [self.num_cross_layers, self.num_cross_layers // 2], "initialization not implemented"

        if config['text_encoder'] == 'data/roberta-base':
            from models.xroberta import RobertaConfig
            config_dec = RobertaConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        else:
            config_dec = copy.deepcopy(config_enc)

        config_dec.encoder_width = config_enc.hidden_size
        config_dec.fusion_layer = 0  # start index
        config_dec.num_hidden_layers = config['num_dec_layers']
        self.cross_encoder_width = config_enc.encoder_width  # i.e. vision_width
        self.dec_encoder_width = config_enc.hidden_size

        if config['text_encoder'] == 'data/roberta-base':
            from models.xroberta import RobertaForCausalLM
            self.text_decoder = RobertaForCausalLM(config=config_dec)
        else:
            self.text_decoder = BertLMHeadModel(config=config_dec)

        if config.get('large_lr_for_dec', False):
            self.init_params = ['text_decoder.' + n for n, _ in self.text_decoder.named_parameters()]
        else:
            if self.dec_encoder_width != self.cross_encoder_width:
                self.init_params = ['text_decoder.' + n for n, _ in self.text_decoder.named_parameters()
                                    if ('crossattention.self.key' in n) or ('crossattention.self.value' in n)]
            else:
                self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(self, ckpt_rpath, config, load_text=False)

            print("### Loading pretrained text encoder", flush=True)
            for key in list(state_dict.keys()):

                name_to_replace = 'bert.'

                if 'roberta' in config['text_encoder']:
                    name_to_replace = 'roberta.'

                if name_to_replace in key:
                    encoder_key = key.replace(name_to_replace, '')
                    state_dict[encoder_key] = state_dict[key]

                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'text_encoder.' in key:
                    if 'layer.' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < self.num_text_layers:
                            del state_dict[key]
                            continue

                        elif (config['num_dec_layers'] == self.num_cross_layers // 2) and (layer_num % 2 == 0):  # 这里加了
                            del state_dict[key]
                            continue

                        elif (self.dec_encoder_width != self.cross_encoder_width) and \
                                (('crossattention.self.key' in key) or ('crossattention.self.value' in key)):
                            del state_dict[key]
                            continue

                        else:

                            if config['num_dec_layers'] == self.num_cross_layers // 2:  # 这里加上, 上面首先删掉了偶数层。
                                decoder_layer_num = (layer_num - self.num_text_layers) // 2
                            elif config['num_dec_layers'] == self.num_cross_layers:
                                decoder_layer_num = (layer_num - self.num_text_layers)
                            else:
                                raise ValueError

                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key

                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, quesiton, answer=None, k=None, weights=None, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)

        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.pad_token_id, -100)

            question_output = self.text_encoder(quesiton.input_ids,
                                                attention_mask=quesiton.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [quesiton.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        else:
            question_output = self.text_encoder(quesiton.input_ids,
                                                attention_mask=quesiton.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask,
                                                    answer.input_ids, answer.attention_mask, k)
            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs


class XVLMPlusForVQA(XVLMPlusBase):
    """
    Generative Model, but model the task as ranking at inference
    """
    def __init__(self, config, tied=False):
        super().__init__(config, load_vision_params=False, load_text_params=False, load_cross_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False)

        self.pad_token_id = build_tokenizer(config['text_encoder']).pad_token_id
        self.text_decoder = self.build_text_decoder(config)
        self.tied = tied
        if self.tied:
            self.tie_enc_and_dec_wordemb()

    def build_text_decoder(self, config):
        assert config['num_dec_layers'] == self.num_cross_layers, "initialization not implemented"
        config_cross = self.cross_encoder.config
        config_dec = copy.deepcopy(config_cross)
        config_dec.encoder_width = config_cross.hidden_size
        config_dec.fusion_layer = 0  # start index
        config_dec.num_hidden_layers = config['num_dec_layers']

        if config['cross_encoder'] == 'data/roberta-base':
            raise NotImplementedError
        else:
            text_decoder = BertLMHeadModel(config=config_dec)

        # set attrs
        self.cross_encoder_width = config_cross.encoder_width  # i.e. vision_width
        self.dec_encoder_width = config_cross.hidden_size

        return text_decoder

    def tie_enc_and_dec_wordemb(self):
        # tie word embeddings
        self.text_decoder.bert.embeddings.word_embeddings.weight = self.text_encoder.embeddings.word_embeddings.weight
        # self.text_decoder.bert.embeddings.position_embeddings.weight = self.text_encoder.embeddings.position_embeddings.weight
        # self.text_decoder.bert.embeddings.token_type_embeddings.weight = self.text_encoder.embeddings.token_type_embeddings.weight

        # tie
        self.text_decoder.cls.predictions.decoder.weight = self.text_decoder.bert.embeddings.word_embeddings.weight

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(self, ckpt_rpath, config, load_text=False)

            print("### Loading pretrained text encoder", flush=True)
            for key in list(state_dict.keys()):
                # XVLMPlus: initialize cross decoder as multimodal encoder
                if 'cross_encoder.' in key:
                    if (self.dec_encoder_width != self.cross_encoder_width) and \
                            (('crossattention.self.key' in key) or ('crossattention.self.value' in key)):
                        continue

                    if 'encoder.layer.' in key:
                        decoder_key = key.replace('cross_encoder.', 'text_decoder.bert.')
                    else:
                        decoder_key = key.replace('cross_encoder.', 'text_decoder.')

                    state_dict[decoder_key] = state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        missing_keys = []
        for p in msg.missing_keys:
            if 'vision_encoder' in p:
                continue

            if self.tied:
                if p == 'text_decoder.bert.embeddings.word_embeddings.weight':  # tied
                    continue

                if p == 'text_decoder.cls.predictions.decoder.weight':  # tied
                    continue

            missing_keys.append(p)

        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

        self.update_init_params(missing_keys)

    def forward(self, image, quesiton, answer=None, k=None, weights=None, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)

        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.pad_token_id, -100)

            question_output = self.get_cross_embeds(image_embeds, image_atts, text_ids=quesiton.input_ids, text_atts=quesiton.attention_mask)

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output[b]] * n
                question_atts += [quesiton.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        else:
            question_output = self.get_cross_embeds(image_embeds, image_atts, text_ids=quesiton.input_ids,
                                                          text_atts=quesiton.attention_mask)

            topk_ids, topk_probs = self.rank_answer(question_output, quesiton.attention_mask,
                                                    answer.input_ids, answer.attention_mask, k)
            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

