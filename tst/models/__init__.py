# flake8: noqa F401, F403
from .resnet_mbn import *
from .wideresnet import *

from .registry import BACKBONES

__all__ = [k for k in globals().keys() if not k.startswith("_")]


def build_backbone(obj_type, *args, **kwargs):
    return BACKBONES.get(obj_type)(*args, **kwargs)
