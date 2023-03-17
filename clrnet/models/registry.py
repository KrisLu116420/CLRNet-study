from clrnet.utils import Registry, build_from_cfg
import torch.nn as nn

BACKBONES = Registry('backbones')   # self.name='backbones'   self._module_dict=dict()
AGGREGATORS = Registry('aggregators')
HEADS = Registry('heads')
NECKS = Registry('necks')
NETS = Registry('nets')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))
#     backbone = dict(
#         type='ResNetWrapper',
#         resnet='resnet101',
#         pretrained=True,
#         replace_stride_with_dilation=[False, False, False],
#         out_conv=False,
#     )

def build_necks(cfg):
    return build(cfg.necks, NECKS, default_args=dict(cfg=cfg))


def build_aggregator(cfg):
    return build(cfg.aggregator, AGGREGATORS, default_args=dict(cfg=cfg))


def build_heads(cfg):
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))
#     heads = dict(type='CLRHead',
#                  num_priors=192,
#                  refine_layers=3,
#                  fc_hidden_dim=64,
#                  sample_points=36)

def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))
#     net = dict(type='Detector', )

def build_necks(cfg):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))
#     neck = dict(type='FPN',
#                 in_channels=[512, 1024, 2048],
#                 out_channels=64,
#                 num_outs=3,
#                 attention=False)
