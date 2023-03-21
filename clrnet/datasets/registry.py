from clrnet.utils import Registry, build_from_cfg

import torch
from functools import partial
import numpy as np
import random
from mmcv.parallel import collate

DATASETS = Registry('datasets')
PROCESS = Registry('process')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_dataset(split_cfg, cfg):
    return build(split_cfg, DATASETS, default_args=dict(cfg=cfg))


def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(split_cfg, cfg, is_train=True):
    if is_train:
        shuffle = True
    else:
        shuffle = False

    dataset = build_dataset(split_cfg, cfg)

    init_fn = partial(worker_init_fn, seed=cfg.seed)  # 根据cfg.seed定义的random seed的函数，每个worker_id有一个相应的seed

    samples_per_gpu = cfg.batch_size // cfg.gpus
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,   # 为True时会在每个epoch重新打乱数据
        num_workers=cfg.workers,   # 用多少个子进程加载数据。0表示数据将在主进程中加载
        pin_memory=False,   # 设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
        drop_last=False,   # 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),   # 将一个batch的数据和标签进行合并操作。
        worker_init_fn=init_fn)   #  如果不是None，在seeding之后，数据加载之前，每个worker子进程将会调用这个函数，并把worker_id作为输入。

    return data_loader
