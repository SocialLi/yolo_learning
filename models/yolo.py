import torch
from torch import nn
import numpy as np
from backbone import build_resnet
from .basic import Conv, SPP
from .loss import compute_loss
from numpy.typing import NDArray


class MyYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01,
                 nms_thresh=0.5) -> None:
        super().__init__()
        self.device = device  # cuda或者是cpu
        self.num_classes = num_classes  # 类别的数量
        self.trainable = trainable  # 训练的标记
        self.conf_thresh = conf_thresh  # 得分阈值
        self.nms_thresh = nms_thresh  # NMS阈值
        self.stride = 32  # 网格的最大步长
        self.grid_cell = self.create_grid(input_size)  # 网格坐标矩阵
        self.input_size = input_size  # 输入图像大小

        # backbone: resnet18
        # feat_dim 为backbone最后输出特征图的通道数
        self.backbone, feat_dim = build_resnet('resnet18', pretrained=trainable)

        # neck: SPP
        self.neck = nn.Sequential(
            SPP(),
            # SPP之后通道数乘4,再接一个1x1的卷积层（conv1x1+BN+LeakyReLU）将通道压缩一下
            Conv(feat_dim * 4, feat_dim, k=1)
        )

        # detection head
        # 直接是4层卷积，用非常简单的1x1卷积和3x3卷积重复堆叠的方式，最后输出特征图的大小和通道数不变
        self.convsets = nn.Sequential(
            Conv(feat_dim, feat_dim // 2, k=1),
            Conv(feat_dim // 2, feat_dim, k=3, p=1),
            Conv(feat_dim, feat_dim // 2, k=1),
            Conv(feat_dim // 2, feat_dim, k=3, p=1)
        )

        # pred
        # 用1x1卷积（不接BN层，不接激活函数）去得到一个13x13x(1+C+4)的特征图
        # 1对应YOLO中的objectness预测，C对应类别预测（PASCAL VOC上，C=20；COCO上，C=80），4则是bbox预测，这里，每个grid处之预测一个bbox，而不是B个
        self.pred = nn.Conv2d(feat_dim, 1 + self.num_classes + 4, kernel_size=1)

        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.1
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :1], bias_value)
        nn.init.constant_(self.pred.bias[..., 1:1 + self.num_classes], bias_value)

    def create_grid(self, input_size):
        """
        用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        这一tensor将在获得边界框参数的时候会会用到

        为了预测中心位置，我们需要获得gridx,gridy，因为YOLOv1是在每个网格预测一个偏移量，所在网格的坐标加上偏移量才是最终的预测结果.
        一个很直接的方法在输出的特征上遍历每一个位置，对每一个位置上的偏移量进行处理即可。但是，这效率不高。
        遵循“能矩阵操作的绝不for循环”个人原则:
            在初始化网络的时候，先生成一个 [1,HxW,2] 的grid_cell矩阵G。
            于是，当我们得到了 [B,HxW,4] 的txtytwth预测，仅需一行代码即可处理得到bbox 的中心点坐标和宽高：
                center_xy = (txtytwth[:,:,:2] + G) x stride
                bbox_wh = e^{txtytwth[:,:,2:]}
            之所上面的中心带你计算中要乘以stride，是因为我们这些操作都是在最后输出的特征图上进行的操作，而不是原图的尺度，因为要乘以stride映射会到输入图像的尺度.
        """
        # 输入图像的高和宽
        w, h = input_size, input_size
        # 特征图的宽和高
        ws, hs = w // self.stride, h // self.stride
        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)], indexing='ij')

        # 将xy两部分的坐标拼起来： [H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)

        return grid_xy

    def set_grid(self, input_size):
        """
            用于重置G矩阵。
        """
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)

    def decode_boxes(self, pred):
        """
        将网络输出的 tx,ty,tw,th 四个参数转换成bbox的 (x1,y1),(x2,y2)
        用到 create_grid 生成的G矩阵。

        a) 由于偏移量c_x,c_y是介于0-1范围内的数，因此，其本身就是有上下界的，而线性输出并没有上下界，这就容易导致在学习的初期，网络可能预测的值非常大，导致bbox分支学习不稳定。
        因此，对于偏移量部分，我们使用 sigmoid 来输出

        b) 边界框的宽高显然是个非负数，而线性输出不能保证这一点，输出一个负数，是没有意义的。
            - 一种解决办法是约束输出为非负，如用ReLU函数，但这种办法就会隐含一个约束条件，这并不利于优化，而且ReLU的0区间无法回传梯度；
            - 另一个办法就是使用 exp-log 方法，具体来说，就是将 w,h 用log函数来处理一下： tw=log(w) th = log(h)
            网络去学习 tw,th，由于这两个量的值域是实数全域，没有上下界，因此就无需担心约束条件对优化带来的影响。
            然后，网络对于预测tw,th的使用exp函数即可得到w,h的值：w=exp(tw),h=exp(th)
        """
        output = torch.zeros_like(pred)
        # 得到所有bbox 的中心点坐标和宽高
        pred[..., :2] = torch.sigmoid(pred[..., :2]) + self.grid_cell
        pred[..., 2:] = torch.exp(pred[..., 2:])

        # 将所有bbox的中心坐标和高宽换算成x1,y1,x2,y2的形式
        output[..., :2] = pred[..., :2] * self.stride - pred[..., 2:] * 0.5
        output[..., 2:] = pred[..., :2] * self.stride + pred[..., 2:] * 0.5

        return output

    def nms(self, bboxes, scores):
        """Pure Python NMS baseline."""
        x1 = bboxes[:, 0]  # xmin
        y1 = bboxes[:, 1]  # ymin
        x2 = bboxes[:, 2]  # xmax
        y2 = bboxes[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order: torch.Tensor = torch.argsort(scores, dim=-1, descending=True)  # 按照降序对bbox的得分进行排序

        keep = []  # 用于保存经过筛选的最终bbox结果
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            # 计算当前bbox（序号为order[0]）与剩下bbox的iou
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 滤除超过nms阈值的检测框
            indexes = np.where(iou <= self.nms_thresh)[0]
            order = order[indexes + 1]

        return keep

    def postprocess(self, bboxes: NDArray, scores: NDArray):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]

        当我们对输出的txtytwth预测处理成bbox的x1y1x2y2后，我们还需要对预测结果进行一次后处理，后处理的主要作用是：
            a) 滤掉那些得分很低的边界框。
            b) 滤掉那些针对同一目标的冗余检测，即非极大值抑制（NMS）处理。

        经过后处理后，我们得到了最终可以输出的三个检测：
            a) bboxes：包含每一个检测框的x1,y1,x2,y2坐标；
            b) scores：包含每一个检测框的得分；
            c) cls_idxs：包含每一个检测框的预测类别序号。
        """
        labels = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), labels]

        # threshold
        # 首先进行阈值筛选，滤除那些得分低的检测框
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        # 对每一类去进行NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(self.num_classes):
            idxs = np.where(labels == i)[0]
            if len(idxs) == 0:
                continue
            c_bboxes = bboxes[idxs]
            c_scores = scores[idxs]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[idxs[c_keep]] = 1

        # 获得最终的检测结果
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        """
        objectness分支我们使用sigmoid来输出，class分支则用softmax来输出
        """

        # backbone主干网络
        feat = self.backbone(x)

        # neck网络
        feat = self.neck(feat)

        # detection head网络
        feat = self.convsets(feat)

        # 预测层
        pred = self.pred(feat)

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred = pred.permute(0, 2, 3, 1).contiguous().flatten(start_dim=1, end_dim=2)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W, 1]
        conf_pred = pred[..., :1]
        # [B, H*W, num_cls]
        cls_pred = pred[..., 1:1 + self.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = pred[..., 1 + self.num_classes:]

        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        conf_pred = conf_pred[0]  # [H*W, 1]
        cls_pred = cls_pred[0]  # [H*W, NC]
        txtytwth_pred = txtytwth_pred[0]  # [H*W, 4]

        # 每个边框的得分
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # 计算边界框，并归一化边界框: [H*W, 4]
        # 对txtytwth_pred变量进行处理得到x1y1x2y2后，又将其除以输入图片的大小做归一化，
        # 同时我们还调用了torch.clamp函数来保证归一化的结果都是01之间的数，这一保证措施相当于我们将那些尺寸大于图片的框进行了剪裁和尺寸为负数的无效的框进行了滤除。
        # 另外，归一化的作用也是有便于后续我们将bbox的坐标从输入图像映射回原始图像上。
        bboxes = self.decode_boxes(txtytwth_pred) / self.input_size
        bboxes = torch.clamp(bboxes, 0., 1.)

        # 将预测放在cpu处理上，以便进行后处理
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone主干网络
            feat = self.backbone(x)

            # neck网络
            feat = self.neck(feat)

            # detection head网络
            feat = self.convsets(feat)

            # 预测层
            pred: torch.Tensor = self.pred(feat)

            # 对 pred 的size做一些view调整，便于后续处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            pred = pred.permute(0, 2, 3, 1).contiguous().flatten(start_dim=1, end_dim=2)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
            # [B, H*W, 1]
            conf_pred = pred[..., 1]
            # [B, H*W, num_cls]
            cls_pred = pred[..., 1:1 + self.num_classes]
            # [B, H*W, 4]
            txtytwth_pred = pred[..., 1 + self.num_classes:]

            # 计算损失
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                total_loss
            ) = compute_loss(pred_conf=conf_pred, pred_cls=cls_pred, pred_txtytwth=txtytwth_pred, targets=targets)

            return conf_loss, cls_loss, bbox_loss, total_loss


yolo = MyYOLO('cuda:0', 512)
