# flake8: noqa F401, F403
from .datasets.cifar import CIFAR10, CIFAR100
from .datasets.svhn import SVHN

# from .datasets.stl10 import STL10
from .datasets.mini_imagenet import MiniImageNet

from .transforms import fixmatch_transform, tst_transform
from .registry import DATASETS
from .registry import TRANSFORMS


__all__ = [k for k in globals().keys() if not k.startswith("_")]


def build_dataset(obj_type, *args, **kwargs):
    return DATASETS.get(obj_type)(*args, **kwargs)


def build_transform(obj_type, *args, **kwargs):
    return TRANSFORMS.get(obj_type)(*args, **kwargs)
