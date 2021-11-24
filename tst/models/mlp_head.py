import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)


def get_mlp_head(mlp_name, input_dim, hidden_dim, low_dim, norm_layer, predictor=False):
    if mlp_name == "moco":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim),
            Normalize(),
        )
    elif mlp_name == "simclr":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim),
            norm_layer(low_dim),
            Normalize(),
        )
    elif mlp_name == "vanilla":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim),
            norm_layer(low_dim),
            Normalize(),
        )
    elif mlp_name == "byol":
        if predictor:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim),
                nn.Linear(low_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim),
                Normalize(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim),
                Normalize(),
            )
    elif mlp_name == "simsiam":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim),
            norm_layer(low_dim),
        )
    elif mlp_name == "classification":
        cls_fc = nn.Linear(input_dim, low_dim)
        cls_fc.weight.data.normal_(0, 0.01)
        cls_fc.bias.data.fill_(0.0)
        return cls_fc
    elif mlp_name == "none":
        return None
    else:
        raise NotImplementedError("MLP version is wrong!")
