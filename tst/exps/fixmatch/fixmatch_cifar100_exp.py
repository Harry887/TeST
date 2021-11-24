import torch
import math
import numpy as np

from torchvision import datasets
from tst.exps.fixmatch_cifar10_exp import Exp as BaseExp
from tst.data import build_dataset, build_transform
from tst.data.sampler import InfiniteSampler
from tst.utils.torch_dist import is_distributed
import tst.utils.torch_dist as dist
from torch.utils.data import DataLoader


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=1024):
        super(Exp, self).__init__(batch_size, max_epoch=max_epoch)
        # ------------------------------------ data loader config ------------------------- #
        self.encoder_arch = "wideresnet"
        self.depth = 28
        self.widen_factor = 8

        self.dataset = "CIFAR100"
        self.dataset_root = "./data"
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        self.transform = "fixmatch_transform"
        self.num_classes = 100
        self.image_size = 32
        self.cutout_size = 32
        self.num_labeled = 2500

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
        base_dataset = datasets.CIFAR100(self.dataset_root, train=True, download=False)
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
        test_dataset = datasets.CIFAR100(self.dataset_root, train=False, transform=transform_val)

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
