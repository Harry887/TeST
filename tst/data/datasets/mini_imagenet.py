import os
import pickle
import torch

from PIL import Image
from ..registry import DATASETS


@DATASETS.register()
class MiniImageNet(torch.utils.data.Dataset):
    base_folder = "mini_imagenet"

    def __init__(self, root, indexs=None, train=True, transform=None, target_transform=None, return_index=None):
        super(MiniImageNet, self).__init__()
        self.root = root
        self.indexs = indexs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index

        # now load the picked numpy arrays
        file_name = "train.pkl" if self.train else "test.pkl"
        file_path = os.path.join(self.root, self.base_folder, file_name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f)
            self.data = entry["image_data"]
            self.targets = entry["image_labels"]
            self.classes = entry["class_names"].values()
            self.idx_to_class = entry["class_names"]

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]

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

    def __len__(self):
        return len(self.data)
