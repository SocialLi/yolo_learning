import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            y: (Tensor) -> [B, 4C, H, W]
        """
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        y = torch.cat([x, x_1, x_2, x_3], dim=1)

        return y