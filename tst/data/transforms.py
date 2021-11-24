import random
import numpy as np
import torchvision.transforms as transforms
import PIL

from PIL import Image, ImageFilter, ImageOps
from .registry import TRANSFORMS

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")


class Solarization(object):
    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, 0),
        (TranslateX, 0.3, 0),
        (TranslateY, 0.3, 0),
    ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m, **kwarg):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.cutout_size = kwarg.get("cutout_size", None)
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        w, h = img.size
        cutout_size = self.cutout_size if self.cutout_size is not None else min(w, h)
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(cutout_size * 0.5))
        return img


class TransformFixMatch(object):
    def __init__(self, image_size, mean, std, cutout_size):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode="reflect"),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode="reflect"),
                RandAugmentMC(n=2, m=10, cutout_size=cutout_size),
            ]
        )
        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        return self.normalize(self.weak(x)), self.normalize(self.strong(x))


@TRANSFORMS.register()
def fixmatch_transform(image_size=32, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616), cutout_size=32):
    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    transform_unlabeled = TransformFixMatch(image_size, mean=mean, std=std, cutout_size=cutout_size)
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transform_labeled, transform_unlabeled, transform_val


class TransformTeST(object):
    def __init__(self, image_size, mean, std, cutout_list=None):
        assert isinstance(cutout_list, list), f"cutout_list expect list type, but got {type(cutout_list)}"
        num_views = len(cutout_list)
        assert num_views >= 1
        self.aug_list = list()
        weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode="reflect"),
            ]
        )
        self.aug_list.append(weak)
        for i in range(num_views):
            strong = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode="reflect"),
                    RandAugmentMC(n=2, m=10, image_size=cutout_list[i]),
                ]
            )
            self.aug_list.append(strong)
        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        return tuple([self.normalize(aug(x)) for aug in self.aug_list])


@TRANSFORMS.register()
def tst_transform(image_size=32, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616), cutout_size=[32]):
    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    transform_unlabeled = TransformTeST(image_size, mean=mean, std=std, cutout_list=cutout_size)
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transform_labeled, transform_unlabeled, transform_val
