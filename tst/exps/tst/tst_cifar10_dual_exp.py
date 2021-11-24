import os
import tst.utils.torch_dist as dist
from tst.models.tst import TeST
from tst.exps.fixmatch.fixmatch_cifar10_exp import Exp as FixMatchExp


class Exp(FixMatchExp):
    def __init__(self, batch_size, max_epoch=1024):
        super(Exp, self).__init__(batch_size, max_epoch)
        self.dataset_root = "data"
        self.transform = "tst_transform"
        self.warmup = 10
        self.window = 10
        self.threshold = 0.95
        self.threshold_lb = 0
        self.threshold_ub = 0.99
        self.threshold_s = 0.7
        self.lambda_u = 10
        self.recall = True
        self.match_low_quality = True
        self.cutout_size = [32, 32, 32]
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        if "model" in self.__dict__:
            return self.model

        model = TeST(
            num_classes=self.num_classes,
            encoder_arch=self.encoder_arch,
            depth=self.depth,
            widen_factor=self.widen_factor,
            drop_rate=0,
            mu=self.mu,
            T=self.temperature,
            eval_step=self.eval_step,
            max_epoch=self.max_epoch,
            num_data=self.num_data,
            batch_size=self.batch_size // dist.get_world_size(),
            lambda_u=self.lambda_u,
            threshold=self.threshold,
            threshold_lb=self.threshold_lb,
            threshold_ub=self.threshold_ub,
            threshold_s=self.threshold_s,
            warmup=self.warmup,
            window=self.window,
            recall=self.recall,
            match_low_quality=self.match_low_quality,
        )
        self.model = model

        return self.model
