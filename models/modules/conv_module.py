import torch
import torch.nn as nn


class ConvModule(nn.Module):

    def __init__(self,
            conv,
            bn,
            act
        ):
        super().__init__()

        self.conv = conv
        self.bn = bn
        self.act = act

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))