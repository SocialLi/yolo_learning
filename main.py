import random

import torch
import numpy as np
import torch.nn as nn
import os.path as osp
import cv2
from models.yolo import MyYOLO
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor


def write_log():
    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs")

    # 绘制 y = 2x 实例
    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)

    # 关闭
    writer.close()


def add_graph():
    device = torch.device('cuda:0')
    model = MyYOLO(device, 512).to(device)
    img = torch.rand([1, 3, 512, 512]).to(device)
    writer = SummaryWriter("logs/graph")
    writer.add_graph(model, input_to_model=img)
    writer.close()


def test():
    device = torch.device('cuda:0')
    model = MyYOLO(device, 512, trainable=False).to(device)
    print(model)
    model.eval()
    img = torch.rand([1, 3, 512, 512]).to(device)
    bboxes, scores, labels = model(img)
    print(bboxes.shape)
    print("========")
    print(scores.shape)
    print("========")
    print(labels.shape)


@torch.no_grad()
def cal(x):
    y = 5 * x
    return y


if __name__ == '__main__':
    a = torch.arange(12).reshape(3, 4)
    print(a)
    b = [aa.tolist() for aa in a]
    print(b)
    # print(z.requires_grad)
    # b = cal(x)
    # print(b.requires_grad)
    # argsort = torch.argsort(a, dim=-1)
    # li = []
    # li.append(argsort[0].item())
    # print(argsort)
    # print(li)
    # test()
    # add_graph()
