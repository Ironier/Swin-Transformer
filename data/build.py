# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import ToNumpy
from typing_extensions import Concatenate
import albumentations as A

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler
from .gid_dataset import GIDDATASET
from .landslide_dataset import LandslideDataset
from .sar_dataset import SARDataset

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

def build_test_loader(config):
    config.defrost()
    dataset_test, config.MODEL.NUM_CLASSES = build_dataset(is_train=False, config=config, is_test=True)
    config.freeze()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank=0
        world_size=1
    if config.TEST.SEQUENTIAL or world_size<=1:
        sampler_val = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=config.TEST.SHUFFLE
        )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return data_loader_val

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank=0
        world_size=1
    print(f"local rank {config.LOCAL_RANK} / global rank {rank} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {rank} successfully build val dataset")

    num_tasks = world_size
    global_rank = rank
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(rank, len(dataset_train), world_size)
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL or world_size<=1:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

class MultiTransform():
    '''
        To do the same enhance simultaneously for image and label.
        args:
            transform_list: [[tranform_function, param_function, param_arg, skip_key]
                            for every transform]
    '''
    def __init__(self, transform_list=[]):
        self.transform_list = transform_list

    def __call__(self, **kwargs):
        #transformed = {}
        #cnt = 0
        for i in self.transform_list:
            args = i[1](*i[2])
            #print(i[0])
            for k,v in kwargs.items():
                if(k in i[3]):
                    continue
                #print(i[0], ' test ', k, type(v))
                kwargs[k] = i[0](v, *args)
                # if(k=='mask'):
                #     cnt +=1
                    #print(cnt,': ',kwargs[k].shape,end='\n')
        return kwargs

    def func_param_none():
        return ()

    def func_param_color_jitter(p):
        return (torch.randperm(4),p,p,p,p)

    def func_param_probability(p)->bool:
        return torch.rand(1)<p

    def func_param_random_crop(input_size, output_size,
                padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        w, h = input_size
        th, tw = output_size
        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )
        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return input_size, padding, pad_if_needed, fill, padding_mode, i, j, th, tw

    def func_param_const(*args):
        return args


    def func_hflip(tensor,flip):
        if(flip):
            return F.hflip(tensor)
        return tensor

    def func_vflip(tensor,flip):
        if(flip):
            return F.vflip(tensor)
        return tensor

    def func_normalize(tensor,mean=None,std=None):
        if mean is None:
            mean = torch.mean(tensor,dim=0,keepdim=True)
        if std is None:
            std = torch.std(tensor,dim=0,keepdim=True)
            std[torch.where(std==0)] = 1
        tensor = F.normalize(tensor,mean,std)
       #tensor[torch.where(tensor==torch.inf)]=0
        return tensor

    def func_color_jitter(img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

    def func_transpose(img):
        return img.permute(1,2,0)

    def func_crop(img, size, padding, pad_if_needed, fill, padding_mode, i, j, h, w):
        if padding is not None:
            img = F.pad(img, padding, fill, padding_mode)
        width, height = F._get_image_size(img)
        # pad the width if needed
        if pad_if_needed and width < size[1]:
            padding = [size[1] - width, 0]
            img = F.pad(img, padding, fill, padding_mode)
        # pad the height if needed
        if pad_if_needed and height < size[0]:
            padding = [0, size[0] - height]
            img = F.pad(img, padding, fill, padding_mode)

        return F.crop(img, i, j, h, w)

def build_dataset(is_train, config, is_test=False):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841

    elif config.DATA.DATASET == 'gid':
        gid_transform=build_transform_for_gid_data(is_train, config)
        if is_test:
            ann_file='test.txt'
        elif is_train:
            ann_file='train.txt'
        else:
            ann_file='val.txt'
        dataset = GIDDATASET(config.DATA.DATA_PATH, ann_file, gid_transform)
        nb_classes = 15

    elif config.DATA.DATASET == 'landslide':
        transform = MultiTransform([]) # 一定要创建个新的对象， 不然transform对象会混淆成一个
        transform.transform_list += [[transforms.ToTensor(),
                                    MultiTransform.func_param_none,(),()]]
        transform.transform_list += [[transforms.Resize(config.DATA.IMG_SIZE, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                                    MultiTransform.func_param_none,(),()]]
        if is_train:
            ann_directory=os.path.join('ImageSets','Segmentation','train')
            transform.transform_list += [[MultiTransform.func_hflip,
                                        MultiTransform.func_param_probability,
                                        (config.AUG.REPROB,),()]]
            transform.transform_list += [[MultiTransform.func_vflip,
                                        MultiTransform.func_param_probability,
                                        (config.AUG.REPROB,),()]]
        else:
            if is_test:
                ann_directory=os.path.join('ImageSets','Segmentation','test')
            else:
                ann_directory=os.path.join('ImageSets','Segmentation','val')
        transform.transform_list += [[MultiTransform.func_normalize,
                                    MultiTransform.func_param_none,(),('mask',)]] #skip mask
        transform.transform_list += [[MultiTransform.func_normalize,
                                    MultiTransform.func_param_const,(-40.8,190.8),('image',)]] #skip image
        dataset = LandslideDataset(config.DATA.DATA_PATH, ann_directory, transform)
        nb_classes=-1

    elif config.DATA.DATASET == 'sar':
        transform = MultiTransform([]) # 一定要创建个新的对象， 不然transform对象会混淆成一个
        transform.transform_list += [[transforms.ToTensor(),
                                    MultiTransform.func_param_none,(),()]]
        transform.transform_list += [[transforms.Resize(config.DATA.IMG_SIZE, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                                    MultiTransform.func_param_none,(),()]]
        transform.transform_list += [[MultiTransform.func_normalize,
                                    MultiTransform.func_param_const,(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),('sar_images')]] #skip sar
        if is_train:
            ann_directory='train'
            transform.transform_list += [[MultiTransform.func_hflip,
                                        MultiTransform.func_param_probability,
                                        (config.AUG.REPROB,),()]]
            transform.transform_list += [[MultiTransform.func_vflip,
                                        MultiTransform.func_param_probability,
                                        (config.AUG.REPROB,),()]]
        else:
            if is_test:
                ann_directory='test'
            else:
                ann_directory='val'
        dataset = SARDataset(config.DATA.DATA_PATH, ann_directory, transform)
        nb_classes=-1
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes

def build_transform_for_gid_data(is_train, config):
    if(is_train):
        transform = A.Compose([
                    A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE),
                    A.HorizontalFlip(p=config.AUG.REPROB),
                    A.RandomCrop(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE),])
    else:
        transform = A.Compose([
                    A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE),])
    return transform

def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 256
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0  else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT!= 'none' else None,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = 256 #int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )
    t.append(ToNumpy())
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
