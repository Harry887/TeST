from abc import ABCMeta, abstractmethod
import functools
import numpy as np
from tabulate import tabulate
from typing import Dict

import torch
from torch.nn import Module

from tst.layers.lr_scheduler import LRScheduler


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self, batch_size, max_epoch):
        self.batch_size = batch_size
        self.max_epoch = max_epoch

        self.seed = None
        self.output_dir = "outputs"
        self.print_interval = 100
        self.dump_interval = 10
        self.eval_interval = 10
        self.enable_tensorboard = False
        self.dataset = "ImageNet"
        self.transform = "typical_imagenet_transform"
        self.image_size = 224

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(self) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(self, **kwargs) -> LRScheduler:
        pass

    def update(self, options: dict) -> str:
        if options is None:
            return None
        assert isinstance(options, dict)
        msg = ""
        for k, v in options.items():
            if k in self.__dict__:
                old_v = self.__getattribute__(k)
                if not v == old_v:
                    self.__setattr__(k, v)
                    msg = "{}\n'{}' is overriden from '{}' to '{}'".format(msg, k, old_v, v)
            else:
                self.__setattr__(k, v)
                msg = "{}\n'{}' is set to '{}'".format(msg, k, v)
        return msg

    def get_cfg_as_str(self) -> str:
        config_table = []
        for c, v in self.__dict__.items():
            if not isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)):
                if hasattr(v, "__name__"):
                    v = v.__name__
                elif hasattr(v, "__class__"):
                    v = v.__class__
                elif type(v) == functools.partial:
                    v = v.func.__name__
            config_table.append((str(c), str(v)))

        headers = ["config key", "value"]
        config_table = tabulate(config_table, headers, tablefmt="rst")
        return config_table

    def __str__(self):
        return self.get_cfg_as_str()
