import torch
import numpy as np
import torch.nn as nn
import os.path as osp
import cv2

if __name__ == '__main__':

    img = cv2.imread(
        "/home/zeyi/PycharmProjects/yolo_learning/data/dataset/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
    print(type(img))