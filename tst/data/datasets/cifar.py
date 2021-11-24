import numpy as np

from PIL import Image
from torchvision import datasets

from ..registry import DATASETS


@DATASETS.register()
class CIFAR10(datasets.CIFAR10):
    def __init__(
        self, root, indexs=None, train=True, transform=None, target_transform=None, return_index=None, download=False
    ):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_index = return_index

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return index, img, target
        return img, target


@DATASETS.register()
class CIFAR100(datasets.CIFAR100):
    def __init__(
        self, root, indexs, train=True, transform=None, target_transform=None, return_index=None, download=False
    ):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_index = return_index

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index is not None:
            return index, img, target
        return img, target
