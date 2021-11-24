import torch
from torch.nn import Module
import torch.nn.functional as F
from . import build_backbone


class FixMatch(Module):
    def __init__(
        self,
        num_classes,
        *,
        depth=28,
        widen_factor=2,
        drop_rate=0.0,
        mu=7,
        T=1,
        threshold=0.95,
        lambda_u=1,
        batch_size=16,
        encoder_arch="wideresnet",
        **kwargs
    ):
        super(FixMatch, self).__init__()
        self.mu = mu
        self.T = T
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.batch_size = batch_size
        self.num_classes = num_classes
        if encoder_arch == "wideresnet":
            self.encoder_arch = build_backbone(
                encoder_arch, num_classes=num_classes, depth=depth, widen_factor=widen_factor, drop_rate=drop_rate
            )
        elif encoder_arch == "resnet18":
            self.encoder_arch = build_backbone(encoder_arch, low_dim=num_classes)
        elif encoder_arch == "resnet18dropout":
            self.encoder_arch = build_backbone(encoder_arch, low_dim=num_classes, drop_rate=drop_rate)
        else:
            self.encoder_arch = build_backbone(
                encoder_arch, num_classes=num_classes, depth=depth, widen_factor=widen_factor, drop_rate=drop_rate
            )

    @staticmethod
    def interleave(x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @staticmethod
    def de_interleave(x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def forward_train(self, x):
        _, inputs_x, inputs_u_w, inputs_u_s, targets_x = x
        inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * self.mu + 1)
        logits = self.encoder_arch.forward(inputs)
        logits = self.de_interleave(logits, 2 * self.mu + 1)
        logits_x = logits[: self.batch_size]
        logits_u_w, logits_u_s = logits[self.batch_size :].chunk(2)
        del logits

        Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

        pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask).mean()
        extra_dict = {}
        return Lx, Lu, mask, extra_dict

    def forward_test(self, x):
        inputs, targets = x
        outputs = self.encoder_arch.forward(inputs)
        loss = F.cross_entropy(outputs, targets)

        return outputs, loss

    def forward(self, x, is_train=True):
        if is_train:
            return self.forward_train(x)
        else:
            return self.forward_test(x)
