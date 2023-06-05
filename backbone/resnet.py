import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import List
import os

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    删掉了global avgpool层和softmax层，二者在目标检测任务中是不需要的，因此也就无需加载进来。
    """

    def __init__(self, block, layers: List[int], zero_init_residual=False) -> None:
        # 参数block是BasicBlock类或Bottleneck类，分别针对 resnet18 和 resnet50 两个主要分类
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 网络一共包含5个卷积组，每个卷积组中包含1个或多个基本的卷积计算过程（Conv-> BN->ReLU）
        # 每个卷积组中包含1次下采样操作，使特征图大小减半。第1个卷积组只包含1次卷积计算操作，5种典型ResNet结构的第1个卷积组完全相同
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建stage1-4。第2-5个卷积组都包含多个相同的残差单元，在很多代码实现上，通常把第2-5个卷积组分别叫做Stage1、Stage2、Stage3、Stage4
        :param block: BasicBlock类
        :param planes: block中间处理的通道数
        :param blocks: 要生成的BasicBlock个数
        :param stride: stride
        :return:
        """
        downsample = None
        # Stage1不需要下采样，此时stride=1。
        # self.inplanes是每个block的输入通道数，经过上一个block之后，inplanes=planes*expansion，同时作为下一个block的输入通道数
        # 值得注意的是，inplanes经过每个stage的第一个block之后，其指都不再改变（也即每个stage的 第2个-第n个 block的输入通道数都是inplanes=planes*expansion）
        if stride != 1 or self.inplanes != planes * block.expansion:
            # stride不为1,或输入通道数不等于expand后的通道数（其实筛选的就是每个stage的第一个block）
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            c5: (Tensor) -> [B, C, H/32, W/32]
        """
        c1 = self.conv1(x)  # [B, C, H/2, W/2]
        c1 = self.bn1(c1)  # [B, C, H/2, W/2]
        c1 = self.relu(c1)  # [B, C, H/2, W/2]
        c2 = self.maxpool(c1)  # [B, C, H/4, W/4]

        c2 = self.layer1(c2)  # [B, C, H/4, W/4]
        c3 = self.layer2(c2)  # [B, C, H/8, W/8]
        c4 = self.layer3(c3)  # [B, C, H/16, W/16]
        c5 = self.layer4(c4)  # [B, C, H/32, W/32]

        return c5


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if os.path.exists("../weights/resnet18-5c106cde.pth"):
            model.load_state_dict(torch.load('../weights/resnet18-5c106cde.pth'), strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        model.load_state_dict(torch.load('../weights/resnet34-333f7ec4.pth'), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        model.load_state_dict(torch.load('../weights/resnet50-19c8e357.pth'), strict=False)
    return model


def build_resnet(model_name='resnet18', pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained)
        feat_dim = 512
    elif model_name == 'resnet34':
        model = resnet34(pretrained)
        feat_dim = 512
    elif model_name == 'resnet50':
        model = resnet34(pretrained)
        feat_dim = 2048

    return model, feat_dim


if __name__ == '__main__':
    model, feat_dim = build_resnet(model_name='resnet18', pretrained=True)
    x = torch.randn(1, 3, 416, 416)
    y = model(x)
    print(y.shape)
