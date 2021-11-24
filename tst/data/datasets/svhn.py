import os
import numpy as np
import torchvision

from torchvision.datasets.utils import verify_str_arg
from PIL import Image

from ..registry import DATASETS


@DATASETS.register()
class SVHN(torchvision.datasets.SVHN):
    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self, root, indexs=None, split="train", transform=None, target_transform=None, return_index=None, download=False
    ):
        super(SVHN, self).__init__(
            root, split=split, transform=transform, target_transform=target_transform, download=download
        )
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        self.return_index = return_index

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        import scipy.io as sio

        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        self.data = loaded_mat["X"]
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()

        # Convert class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return index, img, target

        return img, target

    def __len__(self):
        return len(self.data)
