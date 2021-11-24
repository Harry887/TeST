import math

import torch
import torch.nn as nn

__all__ = [
    "SimpleBatchNorm2d",
    "SimpleBatchNorm1d",
    "MomentumBatchNorm2d",
    "MomentumBatchNorm1d",
    "get_batchnorm_2d",
    "get_batchnorm_1d",
]


class SimpleBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
            result = super(SimpleBatchNorm2d, self).forward(x)
        return result


class SimpleBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
            result = super(SimpleBatchNorm1d, self).forward(x)
        return result


class MomentumBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=1.0,
        affine=True,
        track_running_stats=True,
        total_iters=100,
    ):
        super(MomentumBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0, 2, 3])
        var = torch.var(x, dim=[0, 2, 3])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * (n / (n - 1)) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :, None, None].detach()) / (
            torch.sqrt(tmp_running_var[None, :, None, None].detach() + self.eps)
        )
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch) * 0.5) * (n / (n - 1))
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x


class MomentumBatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=1.0,
        affine=True,
        track_running_stats=True,
        total_iters=100,
    ):
        super(MomentumBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0])
        var = torch.var(x, dim=[0])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * (n / (n - 1)) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :].detach()) / (torch.sqrt(tmp_running_var[None, :].detach() + self.eps))
        if self.affine:
            x = x * self.weight[None, :] + self.bias[None, :]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch) * 0.5) * (n / (n - 1))
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x


def get_batchnorm_2d(bn):
    if bn == "customized":
        norm_layer = SimpleBatchNorm2d
    elif bn == "vanilla":
        norm_layer = nn.BatchNorm2d
    elif bn == "torchsync":
        norm_layer = nn.SyncBatchNorm
    elif bn == "mbn":
        norm_layer = MomentumBatchNorm2d
    elif bn == "partial":
        norm_layer = PartialBatchNorm2d
    else:
        raise ValueError("bn should be none or 'cvsync' or 'torchsync', got {}".format(bn))

    return norm_layer


class PartialBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(PartialBatchNorm2d, self).__init__(num_features, affine=True, track_running_stats=True)
        self.ignore_model_profiling = True
        self.used_channel = num_features
        self.channel = num_features

    def set_used_channel_by_ratio(self, ratio: float):
        self.used_channel = int(round(self.channel * ratio))

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        y = nn.functional.batch_norm(
            input,
            self.running_mean[: self.used_channel],
            self.running_var[: self.used_channel],
            weight[: self.used_channel],
            bias[: self.used_channel],
            self.training,
            self.momentum,
            self.eps,
        )
        return y


def get_batchnorm_1d(bn):
    if bn == "customized" or bn == "partial":
        norm1d_layer = SimpleBatchNorm1d
    elif bn == "vanilla":
        norm1d_layer = nn.BatchNorm1d
    elif bn == "torchsync":
        norm1d_layer = nn.SyncBatchNorm
    elif bn == "mbn":
        norm1d_layer = MomentumBatchNorm1d
    else:
        raise ValueError("bn should be none or 'cvsync' or 'torchsync', got {}".format(bn))

    return norm1d_layer
