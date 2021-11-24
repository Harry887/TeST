import os
import torch
import math
import numpy as np
import torch.optim as optim

from torchvision import datasets
from tst.exps.base_exp import BaseExp
from tst.data import build_dataset, build_transform
from tst.data.sampler import InfiniteSampler
from tst.layers.lr_scheduler import LRScheduler
from tst.utils.torch_dist import is_distributed
import tst.utils.torch_dist as dist
from torch.utils.data import DataLoader


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=1024):

        # ------------------------------------- model config ------------------------------ #
        self.encoder_arch = "wideresnet"
        self.use_ema = True
        self.ema_decay = 0.999
        self.depth = 28
        self.widen_factor = 2

        # ------------------------------------ data loader config ------------------------- #
        self.dataset = "CIFAR10"
        self.dataset_root = "./data"
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

        self.transform = "fixmatch_transform"
        self.num_classes = 10
        self.image_size = 32
        self.cutout_size = 32
        self.num_labeled = 250
        self.mu = 7
        self.expand_labels = True
        self.data_num_workers = 8
        self.seed = 5

        # ------------------------------------  training config --------------------------- #
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.eval_step = 1024 * 64 // batch_size

        self.lambda_u = 1
        self.temperature = 1
        self.threshold = 0.95

        self.warmup_epochs = 0
        self.weight_decay = 5e-4
        self.basic_lr_per_img = 0.03 / 64.0
        self.nesterov = True

        self.use_ema = True
        self.resume = ""
        self.no_progress = False
        self.start_epoch = 0

        # ------------------------------------  functional config ------------------------- #
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.out = "outputs"
        self.print_interval = 100
        self.dump_interval = 10
        self.eval_interval = 10
        self.enable_tensorboard = False

    def get_data_loader(self):
        if "data_loader" in self.__dict__:
            return self.data_loader

        batch_size_per_device = self.batch_size // dist.get_world_size()

        def x_u_split(labels):
            if self.seed:
                np.random.seed(self.seed)
            label_per_class = self.num_labeled // self.num_classes
            labels = np.array(labels)
            labeled_idx = []
            unlabeled_idx = np.array(range(len(labels)))
            for i in range(self.num_classes):
                idx = np.where(labels == i)[0]
                idx = np.random.choice(idx, label_per_class, False)
                labeled_idx.extend(idx)
            labeled_idx = np.array(labeled_idx)
            assert len(labeled_idx) == self.num_labeled

            if self.expand_labels or self.num_labeled < batch_size_per_device:
                num_expand_x = math.ceil(batch_size_per_device * self.eval_step / self.num_labeled)
                labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
            np.random.shuffle(labeled_idx)
            return labeled_idx, unlabeled_idx

        transform_labeled, transform_unlabeled, transform_val = build_transform(
            self.transform, self.image_size, self.mean, self.std, self.cutout_size
        )
        base_dataset = datasets.CIFAR10(self.dataset_root, train=True, download=False)
        labeled_idxs, unlabeled_idxs = x_u_split(base_dataset.targets)
        labeled_dataset = build_dataset(
            self.dataset, self.dataset_root, labeled_idxs, train=True, return_index=True, transform=transform_labeled
        )
        unlabeled_dataset = build_dataset(
            self.dataset,
            self.dataset_root,
            unlabeled_idxs,
            train=True,
            return_index=True,
            transform=transform_unlabeled,
        )
        test_dataset = datasets.CIFAR10(self.dataset_root, train=False, transform=transform_val)

        train_sampler_labeled = InfiniteSampler(len(labeled_dataset), shuffle=True, seed=self.seed if self.seed else 0)
        train_sampler_unlabeled = InfiniteSampler(
            len(unlabeled_dataset), shuffle=True, seed=self.seed if self.seed else 0
        )
        labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=train_sampler_labeled,
            batch_size=batch_size_per_device,
            num_workers=self.data_num_workers,
            drop_last=True,
        )
        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler_unlabeled,
            batch_size=batch_size_per_device * self.mu,
            num_workers=self.data_num_workers,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            sampler=torch.utils.data.distributed.DistributedSampler(test_dataset) if is_distributed() else None,
            batch_size=batch_size_per_device,
            num_workers=self.data_num_workers,
        )
        self.num_data = len(unlabeled_dataset)
        self.data_loader = {
            "train_labeled": labeled_trainloader,
            "train_unlabeled": unlabeled_trainloader,
            "eval": test_loader,
        }
        return self.data_loader

    def get_model(self):
        if "model" in self.__dict__:
            return self.model
        from tst.models.fixmatch import FixMatch

        # logger.info(f"Model: WideResNet {depth}x{widen_factor}")
        model = FixMatch(
            num_classes=self.num_classes,
            depth=self.depth,
            widen_factor=self.widen_factor,
            drop_rate=0,
            mu=self.mu,
            T=self.temperature,
            threshold=self.threshold,
            lambda_u=self.lambda_u,
            batch_size=self.batch_size // dist.get_world_size(),
            encoder_arch=self.encoder_arch,
        )
        # logger.info("Total params: {:.2f}M".format(
        #     sum(p.numel() for p in model.parameters())/1e6))
        self.model = model

        return self.model

    def get_optimizer(self):
        if "optimizer" in self.__dict__:
            return self.optimizer
        model = self.get_model()
        no_decay = ["bias", "bn"]
        grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.SGD(
            grouped_parameters, lr=self.basic_lr_per_img * self.batch_size, momentum=0.9, nesterov=self.nesterov
        )
        return self.optimizer

    def get_lr_scheduler(self, **kwargs):
        if "lr_scheduler" in self.__dict__:
            return self.lr_scheduler
        optimizer = self.get_optimizer()

        self.lr_scheduler = LRScheduler(
            "fixmatch_cos",
            optimizer,
            self.basic_lr_per_img * self.batch_size,
            self.eval_step,
            self.max_epoch,
            warmup_epochs=0,
        )

        return self.lr_scheduler
