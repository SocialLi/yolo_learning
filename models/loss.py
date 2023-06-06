import torch
import torch.nn as nn
import numpy as np
from typing import List
from torch import Tensor


class MSEWithLogitsLoss(nn.Module):
    """
   Sigmoid + MESLoss

   正负样本的loss的权重不同，正样本为5,负样本为1
   """

    def __init__(self, ):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()

        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * inputs ** 2

        loss = 5.0 * pos_loss + 1.0 * neg_loss

        return loss


def generate_dxdywh(gt_label, w, h, s):
    """
    计算正样本所需要的数据
    :param gt_label: 边界框数据
    :param w: 输入图像的宽
    :param h: 输入图像的高
    :param s: 下采样倍率
    :return:
    """
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算边界框的中心点
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1e-4 or box_h < 1e-4:
        # print('Not a valid data !!!')
        return False

    # 计算中心点所在的网格坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 计算中心点偏移量和宽高的标签
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)

    # 计算边界框位置参数的损失权重
    # 越大的框，(box_w / w) * (box_h / h)越大（最大不超过1，因为不能比原图大），那么对应的损失权重weight就越小，反之，越小的框，weight就越大，
    # 这个目的是为了让小框的权重大一些，来提升大小框之间的loss，因为大框相对学起来容易些，所以权重可以小一点，小框学起来难一些，所以权重就给的大一点。
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[]):
    # type: (int, int, List[List]) -> Tensor
    """
    :param input_size: 图片经过一些列变换，最终输入网络的大小
    :param stride: 网络的最大下采样倍率
    :param label_lists: 标注信息列表
    :return: Tensor[B, H*W, 1+1+4+1]。
        H和W是输入图片经过stride下采样后的高和宽（最终特征图的高和宽）。
        1+1+4+1，分别是置信度（1）、类别标签（1）、边界框（4）、边界框回归权重（1）
    """
    # 必要的参数
    batch_size = len(label_lists)
    w = input_size
    h = input_size
    ws = w // stride
    hs = h // stride
    s = stride

    # 准备一个变量gt_tensor ，这个变量的空间维度和我们网络输出的tensor的空间维度是一样的，
    # 也就相当于我们准备好了一个网格，然后在合适的网格位置去保存标签信息，相当于在告诉网络：这个网格有目标，那个网格没有目标。
    # 而最后一个维度：1+1+4+1，分别是置信度（1）、类别标签（1）、边界框（4）、边界框回归权重（1）。
    gt_tensor = np.zeros([batch_size, hs, ws, 1 + 1 + 4 + 1])

    # 制作训练标签
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # 对于输入进来的一批数据的标签，我们使用for循环依次去遍历这一批数据中的每个样本的标签，由于每张图像可能会包含M个目标，
            # 因此再使用一个for循环去遍历当前标签数据中的每一个边界框数据，即代码中的变量gt_label。

            gt_class = int(gt_label[-1])

            # 对于每一个变量gt_label，我们调用generate_txtytwth去计算正样本所需要的数据，包括类别标签和回归边界框需要的txtytwth
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight

    gt_tensor = gt_tensor.reshape((batch_size, -1, 1 + 1 + 4 + 1))

    return torch.from_numpy(gt_tensor).float()


def compute_loss(pred_conf, pred_cls, pred_txtytwth, targets):
    """
    计算损失
    :param pred_conf: [B, HW, 1]
    :param pred_cls: [B, HW, num_classes]
    :param pred_txtytwth: [B, HW, 4]
    :param targets:  Tensor[B, H*W, 1+1+4+1]
    :return:
    """
    batch_size = pred_conf.size(0)
    # 损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')  # Sigmoid + BCELoss
    twth_loss_function = nn.MSELoss(reduction='none')

    """
    注意，在上面的loss函数中，我们没有对中心点偏移量使用sigmoid函数，这是因为在PyTorch所提供的 torch.nn.BCEWithLogitsLoss 函数中已经自带了sigmoid功能，
    该函数内部会对给当输入做一次sigmoid处理，无需我们在外部单独做一次sigmoid激活。类别预测没有使用softmax函数处理也是同理.
    """

    # 预测
    pred_conf = pred_conf[:, :, 0]  # [B, HW,]
    pred_cls = pred_cls.permute(0, 2, 1)  # [B, C, HW]
    pred_txty = pred_txtytwth[:, :, :2]  # [B, HW, 2]
    pred_twth = pred_txtytwth[:, :, 2:]  # [B, HW, 2]

    # 标签
    gt_obj = targets[:, :, 0]  # [B, HW,]
    gt_cls = targets[:, :, 1].long()  # [B, HW,]
    gt_txty = targets[:, :, 2:4]  # [B, HW, 2]
    gt_twth = targets[:, :, 4:6]  # [B, HW, 2]
    gt_box_scale_weight = targets[:, :, 6]  # [B, HW,]

    # 置信度损失
    conf_loss = conf_loss_function(pred_conf, gt_obj)
    conf_loss = conf_loss.sum() / batch_size

    # 类别损失
    # 乘以gt_obj是因为只有正样本才计算类别损失，同理下面的边界框损失
    cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_obj
    cls_loss = cls_loss.sum() / batch_size

    # 边界框txty的损失
    txty_loss = txty_loss_function(pred_txty, gt_txty).sum(-1) * gt_obj * gt_box_scale_weight
    txty_loss = txty_loss.sum() / batch_size

    # 边界框twth的损失
    twth_loss = twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_obj * gt_box_scale_weight
    twth_loss = twth_loss.sum() / batch_size

    bbox_loss = txty_loss + twth_loss

    # 总的损失
    total_loss = conf_loss + cls_loss + bbox_loss

    return conf_loss, cls_loss, bbox_loss, total_loss
