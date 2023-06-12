import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import InterpolationMode

from dataset.tokenizers import build_tokenizer
from dataset.pretrain_dataset import ImageTextJsonDataset, RegionTextJsonDataset, TextJsonDataset, FrameTextDataset
from dataset.pretrain_dataset_multilingual import ImageMultiTextDataset, RegionMultiTextDataset, ParaTextDataset

from dataset.retrieval_dataset import re_train_dataset, re_eval_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.vqa_dataset import vqa_dataset, msrvtt_qa_dataset, msvd_qa_dataset, vqa_classify_dataset
from dataset.grounding_dataset import grounding_dataset, grounding_dataset_bbox
from dataset.captioning_dataset import coco_karpathy_train, coco_karpathy_train_mlm, coco_karpathy_train_scst, coco_karpathy_caption_eval

from dataset.vqa_dataset import xgqa_dataset, next_qa_mc_dataset
from dataset.xvnli_dataset import xvnli_dataset
from dataset.xflickrco_dataset import xflickrco_train_dataset, xflickrco_eval_dataset
from dataset.wit_dataset import wit_train_dataset, wit_eval_dataset

from dataset.randaugment import RandomAugment


def create_dataset(dataset, config, evaluate=False):

    if dataset == 'pretrain_text':
        text_dataset = TextJsonDataset(config, config['train_file_text'], rank=int(os.environ.get('RANK') or 0),
                                       world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        return text_dataset

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        if len(config.get('train_file', [])):
            image_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                                   world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                   transform=pretrain_transform)
        else:
            image_dataset = None


        if len(config.get('train_file_regions', [])):
            region_dataset = RegionTextJsonDataset(config, config['train_file_regions'], rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                    transform=pretrain_transform, box_transform=box_transform)
        else:
            region_dataset = None

        if len(config.get('train_file_aux', [])):  # cleaner image-text pairs
            image_dataset_aux = ImageTextJsonDataset(config, config['train_file_aux'], rank=int(os.environ.get('RANK') or 0),
                                                   world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                   repeat=True, transform=pretrain_transform, is_aux=True)
        else:
            image_dataset_aux = None

        if len(config.get('train_file_videos', [])):
            video_dataset = FrameTextDataset(config, config['train_file_videos'], rank=int(os.environ.get('RANK') or 0),
                                                 world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                 repeat=True, transform=pretrain_transform, training=True)
        else:
            video_dataset = None

        if len(config.get('train_file_videos_aux', [])):
            video_dataset_aux = FrameTextDataset(config, config['train_file_videos_aux'], rank=int(os.environ.get('RANK') or 0),
                                                 world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                 repeat=True, transform=pretrain_transform, training=True)
        else:
            video_dataset_aux = None

        if len(config.get('train_file_text', [])):
            text_dataset = TextJsonDataset(config, config['train_file_text'], rank=int(os.environ.get('RANK') or 0),
                                           world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        else:
            text_dataset = None

        if len(config.get('train_file_mtext', [])):  # multilingual parallel texts
            assert len(config.get('train_file_text', [])) == 0
            mtext_dataset = ParaTextDataset(config, config['train_file_mtext'], rank=int(os.environ.get('RANK') or 0),
                                           world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        else:
            mtext_dataset = None

        return image_dataset, region_dataset, text_dataset, image_dataset_aux, video_dataset, video_dataset_aux, mtext_dataset

    elif dataset == 'pretrain_multilingual':
        if len(config['train_file']):
            image_dataset = ImageMultiTextDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                    repeat=True, transform=pretrain_transform)
        else:
            image_dataset = None

        if len(config['train_file_regions']):
            region_dataset = RegionMultiTextDataset(config, config['train_file_regions'],
                                                    rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                    repeat=True, transform=pretrain_transform, box_transform=box_transform)
        else:
            region_dataset = None

        if len(config['train_file_mono']):  # monolingual (e.g. en/zh) x multimodal data
            mono_dataset = ImageTextJsonDataset(config, config['train_file_mono'], rank=int(os.environ.get('RANK') or 0),
                                                 world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                 repeat=True, transform=pretrain_transform, config_key='images_mono')
        else:
            mono_dataset = None

        if len(config['train_file_text']):
            text_dataset = ParaTextDataset(config, config['train_file_text'], rank=int(os.environ.get('RANK') or 0),
                                           world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        else:
            text_dataset = None

        return image_dataset, region_dataset, mono_dataset, text_dataset

    elif dataset == 'infer_xemb':
        return infer_xemb_dataset(config['json_rpath'].strip().split(','), test_transform)

    elif dataset == 're':
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        if evaluate:
            return None, None, test_dataset

        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])

        return train_dataset, val_dataset, test_dataset

    elif dataset == 're_video':
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'],
                                       max_words=config['max_words'],
                                       index_key=config['index_key'], vision_key=config['vision_key'],
                                       text_key=config['text_key'],
                                       is_video=True, frame_len=config['frame_len'])

        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'],
                                       max_words=config['max_words'],
                                       index_key=config['index_key'], vision_key=config['vision_key'],
                                       text_key=config['text_key'],
                                       is_video=True, frame_len=config['frame_len'])

        return train_dataset, test_dataset

    elif dataset == 're_video_dist':
        train_dataset = dist_re_train_dataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                                   world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                   transform=train_transform)

        test_img_dataset = dist_re_eval_dataset(config, config['test_file'], rank=int(os.environ.get('RANK') or 0), mode="image",
                                                   world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=False, repeat=False,
                                                   transform=test_transform)

        test_text_dataset = dist_re_eval_dataset(config, config['test_file'], rank=int(os.environ.get('RANK') or 0), mode="text",
                                                   world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=False, repeat=False,
                                                   transform=test_transform)

        return train_dataset, test_img_dataset, test_text_dataset

    elif dataset == 'vqa_next':
        vqa_test_dataset = next_qa_mc_dataset(config['test_file'], test_transform, config['img_rdir'], 
                                       split='test', text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        valid_dataset = next_qa_mc_dataset(config['valid_file'], test_transform, config['img_rdir'], 
                                       split='test', text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        if evaluate:
            return None, vqa_test_dataset

        train_dataset = next_qa_mc_dataset(config['train_file'],  train_transform_wohflip, config['img_rdir'], 
                                    split='train', text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        return train_dataset, valid_dataset, vqa_test_dataset

    elif dataset == 'xre':
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])

        val_dataset_dict = {}
        for k, rpath in config['val_file'].items():
            val_dataset_dict[k] = re_eval_dataset(rpath, test_transform, config['image_root'])

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = re_eval_dataset(rpath, test_transform, config['image_root'])

        return train_dataset, val_dataset_dict, test_dataset_dict

    elif dataset == 'vqa':
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'],
                                       text_encoder=config['text_encoder'])

        if evaluate:
            return None, vqa_test_dataset

        train_dataset = vqa_dataset(config['train_file'], train_transform_wohflip, config['vqa_root'], config['vg_root'],
                                    split='train', text_encoder=config['text_encoder'])

        return train_dataset, vqa_test_dataset
    
    elif dataset == 'vqa_classify':
        vqa_test_dataset = vqa_classify_dataset(config['test_file'], config['ans2label_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'],
                                       text_encoder=config['text_encoder'])

        if evaluate:
            return None, vqa_test_dataset

        train_dataset = vqa_classify_dataset(config['train_file'], config['ans2label_file'], train_transform_wohflip, config['vqa_root'], config['vg_root'],
                                    split='train', text_encoder=config['text_encoder'])

        return train_dataset, vqa_test_dataset

    elif dataset == 'vqa_msrvtt':
        train_dataset = msrvtt_qa_dataset(config['train_file'], config['ans2label_file'], train_transform_wohflip,
                                          config['img_rdir'], 'train', config['max_tokens'],
                                          text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        valid_dataset = msrvtt_qa_dataset(config['valid_file'], config['ans2label_file'], test_transform,
                                          config['img_rdir'], 'test', config['max_tokens'],
                                          text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        test_dataset = msrvtt_qa_dataset(config['test_file'], config['ans2label_file'], test_transform,
                                         config['img_rdir'], 'test', config['max_tokens'],
                                         text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        return train_dataset, valid_dataset, test_dataset

    elif dataset == 'vqa_msvd':
        train_dataset = msvd_qa_dataset(config['train_file'], config['ans2label_file'], train_transform_wohflip,
                                          config['img_rdir'], 'train', config['max_tokens'],
                                          text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        valid_dataset = msvd_qa_dataset(config['valid_file'], config['ans2label_file'], test_transform,
                                          config['img_rdir'], 'test', config['max_tokens'],
                                          text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        test_dataset = msvd_qa_dataset(config['test_file'], config['ans2label_file'], test_transform,
                                         config['img_rdir'], 'test', config['max_tokens'],
                                         text_encoder=config['text_encoder'], frame_len=config['frame_len'])

        return train_dataset, valid_dataset, test_dataset

    elif dataset == 'xgqa':
        train_dataset = xgqa_dataset(config['train_file'], train_transform_wohflip, config['vqa_root'],
                                    split='train', text_encoder=config['text_encoder'])

        valid_dataset = xgqa_dataset(config['valid_file'], test_transform, config['vqa_root'],
                                    split='test', answer_list=config['answer_list'],
                                    text_encoder=config['text_encoder'])

        test_dataset_dict = {}
        for language, (rpath, ans_rpath) in config['test_file'].items():
            test_dataset_dict[language] = xgqa_dataset(rpath, test_transform, config['vqa_root'], split='test', answer_list=ans_rpath,
                                                      text_encoder=config['text_encoder'])

        return train_dataset, valid_dataset, test_dataset_dict

    elif dataset == 'nlvr':
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        if evaluate:
            return None, None, test_dataset

        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'marvl':
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])

        test_dataset_dict = {}  # marvl test
        for k, rpath in config['test_file'].items():
            if k == 'en':  # marvl does not have en test, so i use nlvr2 test set
                test_dataset_dict[k] = nlvr_dataset(rpath, test_transform, image_root=config['image_root'])
            else:
                test_dataset_dict[k] = nlvr_dataset(rpath, test_transform, image_root=None)

        return train_dataset, val_dataset, test_dataset_dict


    elif dataset == 'xvnli':
        train_dataset = xvnli_dataset(config['train_file'], train_transform, config['image_root'], config['max_tokens'])
        val_dataset = xvnli_dataset(config['val_file'], test_transform, config['image_root'], config['max_tokens'])

        test_dataset_dict = {}  # marvl test
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = xvnli_dataset(rpath, test_transform, config['image_root'], config['max_tokens'])

        return train_dataset, val_dataset, test_dataset_dict

    elif dataset == 'xflickrco':
        train_dataset = xflickrco_train_dataset(config['train_file'], train_transform,
                                                config['image_root']['flickr30k'])

        val_dataset = xflickrco_eval_dataset(config['val_file'], test_transform, config['image_root'])

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = xflickrco_eval_dataset(rpath, test_transform, config['image_root'])

        return train_dataset, val_dataset, test_dataset_dict

    elif dataset == 'wit':
        train_dataset = wit_train_dataset(config['train_file'], train_transform)

        val_dataset = wit_eval_dataset(config['val_file'], test_transform)

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = wit_eval_dataset(rpath, test_transform)

        return train_dataset, val_dataset, test_dataset_dict

    elif dataset == 'grounding_bbox':
        test_dataset = grounding_dataset_bbox(config['test_file'], test_transform, config['image_root'], mode='test', config=config)
        if evaluate:
            return None, test_dataset

        train_transform = transforms.Compose([
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = grounding_dataset_bbox(config['train_file'], train_transform, config['image_root'], mode='train', config=config)
        return train_dataset, test_dataset

    elif dataset == 'captioning_pretrain':
        image_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                               transform=pretrain_transform, add_eos=True)
        return image_dataset

    elif dataset == 'caption_coco':
        train_dataset = coco_karpathy_train(train_transform, config['image_root'], config['train_file'], prompt=config['prompt'], max_words=config['max_tokens'])
        val_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['val_file'], 'val')
        test_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['test_file'], 'test')

        return train_dataset, val_dataset, test_dataset

    elif dataset == 'caption_coco_mlm':
        train_dataset = coco_karpathy_train_mlm(train_transform, config['image_root'], config['train_file'], config)
        val_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['val_file'], 'val')
        test_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['test_file'], 'test')

        return train_dataset, val_dataset, test_dataset

    elif dataset == 'classify':
        image_dataset = classify_dataset(config['train_image_file'], train_transform, rank=int(os.environ.get('RANK') or 0),
                                         world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                         mapper_path=config['cnd_file'], is_test=False)

        test_dataset_dict = {}
        for k, v in config['test_file'].items():
            test_dataset_dict[k] = classify_dataset(v, test_transform, rank=int(os.environ.get('RANK') or 0),
                                        world_size=int(os.environ.get('WORLD_SIZE') or 1),
                                        mapper_path=config['cnd_file'], is_test=True)

        if len(config.get('train_text_file', {})):
            text_dataset = classify_dataset(config['train_text_file'], train_transform, rank=int(os.environ.get('RANK') or 0),
                                             world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                             mapper_path=config['cnd_file'], is_test=False)
        else:
            text_dataset = None

        return image_dataset, test_dataset_dict, text_dataset

    elif dataset == 'translate':
        train_dataset = translate_dataset(config['train_file'], train_transform, image_key=config['image_key'],
                                          src_key=config['src_key'], tgt_key=config['tgt_key'], is_image_rpath=config['is_image_rpath'],
                                          is_test=False)
        test_dataset = translate_dataset(config['test_file'], test_transform, image_key=config['image_key'],
                                          src_key=config['src_key'], tgt_key=config['tgt_key'], is_image_rpath=config['is_image_rpath'],
                                         is_test=True)

        return train_dataset, test_dataset

    elif dataset == 'classify_tns_video_domain_pretrain':
        video_dataset = TNSVideoDataset(config, config['train_file_videos'], rank=int(os.environ.get('RANK') or 0),
                                         world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                         repeat=True, transform=train_transform, training=True)

        video_neg_dataset = TNSVideoDataset(config, config['train_file_videos_neg'], rank=int(os.environ.get('RANK') or 0),
                                         world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                         repeat=True, transform=train_transform, training=True)

        if len(config.get('train_file_regions', [])):
            region_dataset = RegionMultiTextDataset(config, config['train_file_regions'],
                                                    rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                    repeat=True, transform=pretrain_transform, box_transform=box_transform)

        else:
            region_dataset = None

        if len(config.get('test_file_videos', [])):
            video_test_dataset = TNSVideoDataset(config, config['test_file_videos'],
                                                 rank=int(os.environ.get('RANK') or 0),
                                                 world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=False,
                                                 repeat=False, transform=test_transform, training=False)
        else:
            video_test_dataset = None

        return video_dataset, region_dataset, video_test_dataset, video_neg_dataset

    elif dataset == 'classify_tns_profile_domain_pretrain':
        train_dataset = TNSProfileDataset(config, config['train_file'], train_transform, rank=int(os.environ.get('RANK') or 0),
                                         world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)

        if len(config.get('train_file_regions', [])):
            region_dataset = RegionMultiTextDataset(config, config['train_file_regions'],
                                                    rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                    repeat=True, transform=pretrain_transform, box_transform=box_transform)

        else:
            region_dataset = None

        if len(config.get('test_file', [])):
            test_dataset = TNSProfileDataset(config, config['test_file'], test_transform,
                                              rank=int(os.environ.get('RANK') or 0),
                                              world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=False,
                                              repeat=False)

        else:
            test_dataset = None

        if len(config.get('train_file_text', [])):
            text_dataset = ParaTextDataset(config, config['train_file_text'], rank=int(os.environ.get('RANK') or 0),
                                           world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        else:
            text_dataset = None

        return train_dataset, region_dataset, test_dataset, text_dataset

    else:
        raise NotImplementedError(f"dataset == {dataset}")


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def vqa_classify_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights, _ in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, torch.Tensor(answer_list), torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders

def vqa_mc_collate_fn(batch):
    image_list, question_list, answer_list = [], [], []
    cand_list = [ [], [], [], [], [] ]
    for image, question, answer, cand_ in batch:
        image_list.append(image)
        question_list.append(question)     
        answer_list.append(int(answer))
        # cand_list += cand_
        # print(cand_)
        for i in range(5):
            cand_list[i].append(cand_[i])
    return torch.stack(image_list, dim=0), question_list, torch.Tensor(answer_list), cand_list