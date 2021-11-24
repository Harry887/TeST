import math
from functools import partial

__all__ = ["LRScheduler", "cos_lr", "warm_cos_lr", "multistep_lr"]


class LRScheduler:
    def __init__(self, name, optimizer, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Supported lr schedulers: [cos, warmcos, multistep]
        Extra keyword arguments:
          - cos: None
          - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
          - multistep: [milestones (epochs), gamma (default 0.1)]
        """

        self._optimizer = optimizer
        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs

        self.__dict__.update(kwargs)

        self.lr_func = self._get_lr_func(name)

    def update_lr(self, iters):
        lr = self.lr_func(iters)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr_func(self, name):
        if name == "cos":  # cosine lr schedule
            end_lr = getattr(self, "end_lr", 0.0)
            if end_lr > 0:
                lr_func = partial(cos_w_end_lr, self.lr, self.total_iters, end_lr)
            else:
                lr_func = partial(cos_lr, self.lr, self.total_iters)
        elif name == "warmcos":
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
            end_lr = getattr(self, "end_lr", 0.0)
            if end_lr > 0:
                lr_func = partial(
                    warm_cos_w_end_lr, self.lr, self.total_iters, warmup_total_iters, warmup_lr_start, end_lr
                )
            else:
                lr_func = partial(warm_cos_lr, self.lr, self.total_iters, warmup_total_iters, warmup_lr_start)
        elif name == "multistep":  # stepwise lr schedule
            milestones = [int(self.total_iters * milestone / self.total_epochs) for milestone in self.milestones]
            gamma = getattr(self, "gamma", 0.1)
            lr_func = partial(multistep_lr, self.lr, milestones, gamma)
        elif name == "fixmatch_cos":  # cosine lr schedule
            lr_func = partial(fixmatch_cos_lr, self.lr, self.total_iters)
        else:
            raise ValueError("Scheduler version {} not supported.".format(name))
        return lr_func


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr


def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
    else:
        lr *= 0.5 * (1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters)))
    return lr


def multistep_lr(lr, milestones, gamma, iters):
    """MultiStep learning rate"""
    for milestone in milestones:
        lr *= gamma if iters >= milestone else 1.0
    return lr


def warm_cos_w_end_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, end_lr, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
    else:
        q = 0.5 * (1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters)))
        lr = lr * q + end_lr * (1 - q)
    return lr


def cos_w_end_lr(lr, total_iters, end_lr, iters):
    """Cosine learning rate"""
    q = 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    lr = lr * q + end_lr * (1 - q)
    return lr


def fixmatch_cos_lr(lr, total_iters, iters, num_cycles=7.0 / 16.0):
    """FixMatch Cosine learning rate"""
    lr *= max(0, math.cos(math.pi * float(iters) / float(total_iters) * num_cycles))
    return lr


class GroupLRScheduler:
    def __init__(self, name_dict, optimizer, lr_dict, iters_per_epoch, total_epochs, **kwargs):
        """
        Supported lr schedulers: [cos, warmcos, multistep]
        Extra keyword arguments:
          - cos: None
          - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
          - multistep: [milestones (epochs), gamma (default 0.1)]
        """
        assert isinstance(name_dict, dict)
        self._optimizer = optimizer
        self._lr_dict = {}
        self._lr_func_dict = {}

        for name in name_dict:
            new_kwargs = {k: v[name] for k, v in kwargs.items() if name in v and v.get(name) is not None}
            lr_scheduler = LRScheduler(
                name_dict[name], None, lr_dict[name], iters_per_epoch, total_epochs, **new_kwargs
            )
            self._lr_dict[name] = lr_dict[name]
            self._lr_func_dict[name] = lr_scheduler.lr_func

    def update_lr(self, iters):
        for param_group in self._optimizer.param_groups:
            name = param_group["name"]
            lr = self._lr_func_dict[name](iters)
            param_group["lr"] = lr
            self._lr_dict[name] = lr
        return self._lr_dict
