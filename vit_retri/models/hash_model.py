import torch
import torch.nn as nn
import numpy as np

from vit_retri.models import VisionTransformer
from vit_retri.models import CONFIGS


class DSHNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(DSHNet, self).__init__()

        config = CONFIGS[args.model_type]
        self.vit = VisionTransformer(config, args.img_size, zero_head=True, num_classes=200)
        self.vit.load_from(np.load(args.pretrained_dir))
        self.hash_layer = nn.Linear(config.hidden_size, args.hash_bit)

    def forward(self, x):
        feats, _ = self.vit(x)
        y = self.hash_layer(feats)
        return y

