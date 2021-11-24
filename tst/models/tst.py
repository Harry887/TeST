import torch
import torch.nn.functional as F
from tst.models.fixmatch import FixMatch


class TeST(FixMatch):
    def __init__(self, num_classes, **kwargs):
        super(TeST, self).__init__(num_classes, **kwargs)
        self.max_epoch = kwargs.get("max_epoch", 1024)
        self.eval_step = kwargs.get("eval_step", 1024)
        self.num_data = kwargs.get("num_data", 50000)

        self.threshold_lb = kwargs.get("threshold_lb", 0)
        self.threshold_ub = kwargs.get("threshold_ub", 0.99)
        self.threshold_s = kwargs.get("threshold_s", 0.7)
        self.warmup_epochs = kwargs.get("warmup", 0)
        self.windows = kwargs.get("window", 10)
        self.alpha = kwargs.get("alpha", 1)
        self.recall = kwargs.get("recall", False)
        self.match_low_quality = kwargs.get("match_low_quality", False)

        self.epoch_count = 0
        self.iter_count = 0

        self.init_max_probs(kwargs.get("max_probs", None))

    def init_max_probs(self, max_probs=None):
        if max_probs is None:
            self.register_buffer("max_probs", torch.ones((self.num_data, self.windows), dtype=torch.float32))
        else:
            self.register_buffer("max_probs", max_probs)

    def push_table(self, table, idxs, items):
        table[idxs] = torch.cat((table[idxs, 1:], items.reshape(-1, 1)), dim=1)

    def forward_train(self, x):
        self.iter_count += 1

        unlabel_idxs, inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1, inputs_u_s2, targets_x = x
        inputs = self.interleave(
            torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1, inputs_u_s2)), 4 * self.mu + 1
        )
        logits = self.encoder_arch.forward(inputs)
        logits = self.de_interleave(logits, 4 * self.mu + 1)
        logits_x = logits[: self.batch_size]
        logits_u_w, logits_u_s, logits_u_s1, logits_u_s2 = logits[self.batch_size :].chunk(4)
        del logits

        Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

        pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)
        max_probs_u, targets_u = torch.max(pseudo_label, dim=-1)

        # Calculate threshold by mean and std
        max_prob_mean = self.max_probs.mean(dim=1)
        max_prob_std = self.max_probs.std(dim=1)
        thr = torch.clamp((max_prob_mean + self.alpha * max_prob_std), min=self.threshold_lb, max=self.threshold_ub)
        threshold = thr[unlabel_idxs] if self.epoch_count >= self.warmup_epochs else self.threshold
        mask_weak = max_probs_u.ge(threshold)

        self.push_table(self.max_probs, unlabel_idxs, max_probs_u)

        # Multi-crops Consistency Loss
        logits_crops = torch.stack((logits_u_s, logits_u_s1, logits_u_s2), dim=-1)
        pseudo_label_crops = torch.softmax(logits_crops.detach(), dim=1)
        max_probs_crops, targets_crops = torch.max(pseudo_label_crops, dim=1)
        mask_crops = max_probs_crops.ge(self.threshold_s)

        # For each valid pseudo label, assign its most suitable crop
        if self.recall:
            mask_all_neg_crops = mask_crops.sum(dim=1) == 0
            mask_recall = mask_weak & mask_all_neg_crops
            if self.match_low_quality:
                recall_crop_probs, recall_crop_idxs = torch.min(max_probs_crops, dim=1)
            else:
                recall_crop_probs, recall_crop_idxs = torch.max(max_probs_crops, dim=1)
            mask_crops[torch.nonzero(mask_recall).reshape(-1), recall_crop_idxs[mask_recall]] = True
        mask = mask_crops & mask_weak.reshape(-1, 1).repeat(1, 3)

        Lu = (F.cross_entropy(logits_crops, targets_u.reshape(-1, 1).repeat(1, 3), reduction="none") * mask).mean()

        if self.iter_count % self.eval_step == 0:
            self.iter_count = 0
            self.epoch_count += 1

        extra_dict = {
            "max_prob_mean": max_prob_mean.mean(),
            "max_prob_std": max_prob_std.mean(),
            "max_probs_crops": max_probs_crops.mean(),
            "mask_crops": mask_crops.float().mean(),
        }
        return Lx, Lu, mask_weak.float(), extra_dict
