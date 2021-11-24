from argparse import Action
import os
import shutil

import torch
import torchvision

# PyTorch version as a tuple of 2 ints. Useful for comparison.
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
TORCHVISION_VERSION = tuple(int(x) for x in torchvision.__version__.split(".")[:2])


def save_checkpoint(state, is_best, save, model_name=""):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, model_name + "_best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_devices(gpu_ids):
    if "-" in gpu_ids:
        gpus = gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids


class AvgMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()
        self.val = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class DataPrefetcher:
    """
    Attribute:
        data_format (str): "list" indicates list inputs
                           "image" indicates image input.
    """

    def __init__(self, loader, data_format="list"):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_list if data_format == "list" else self._input_cuda_for_image
        self.record_stream = (
            DataPrefetcher._record_stream_for_list if data_format == "list" else DataPrefetcher._record_stream_for_image
        )
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_list(self):
        for indx in range(len(self.next_input)):
            self.next_input[indx] = self.next_input[indx].cuda(non_blocking=True)
            self.next_input[indx] = self.next_input[indx].float()

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)
        self.next_input = self.next_input.float()

    @staticmethod
    def _record_stream_for_list(input):
        for indx in range(len(input)):
            input[indx].record_stream(torch.cuda.current_stream())

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def set_unittest_seed(seed=1):
    import random
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if TORCH_VERSION >= (1, 7):
        torch.set_deterministic(True)


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.
        All elements inside '()' or '[]' are treated as iterable values.
        Args:
            val (str): Value string.
        Returns:
            list | tuple: The expanded list or tuple from the string.
        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.
            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (char == ",") and (pre.count("(") == pre.count(")")) and (pre.count("[") == pre.count("]")):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
