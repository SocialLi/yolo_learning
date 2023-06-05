import argparse
import os
import random
import time

import torch
import torch.utils.data
import torch.nn.functional as F
from torch import optim
from torch.optim import Optimizer
from data.transform import BaseTransform, Augmentation
from data.voc0712 import VOCDetection
from evaluator.vocapi_evaluator import VOCAPIEvaluator
from utils.misc import detection_collate
from utils.com_paras_flops import FLOPs_and_Params
from models.build import build_yolo
from models.loss import gt_creator
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # 基本参数
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False, help='use tensorboard')
    parser.add_argument('--eval_epoch', type=int, default=10, help='interval between evaluations')
    parser.add_argument('--save_folder', type=str, default='weights/', help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')

    # 模型参数
    parser.add_argument('-v', '--version', default='yolo', help='yolo')

    # 训练配置
    parser.add_argument('-bs', '--batch_size', default=8, type=int, help='Batch size for training')
    """
    -accu 4表示我们累计四次梯度再反向传播，假如batch size为16，那我们累计4次再传向传播的话，相当于batch size为16x4=64；
    这里建议读者保持-bs x -accu = 64 的配置，比如，读者设置-bs 8，那么就要相应地设置-accu 8，以确保二者的乘积为64，这里建议乘积不要小于32；
    """
    parser.add_argument('-accu', '--accumulate', default=8, type=int, help='gradient accumulate.')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1, help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, help='keep training')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False, help='use multi-scale trick')
    parser.add_argument('--max_epoch', type=int, default=150, help='Maximum number of iterations')
    parser.add_argument('--lr_epoch', nargs='+', default=[90, 120], type=int, help='lr epoch to decay')

    # 优化器参数
    parser.add_argument('-lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

    # 数据集参数
    parser.add_argument('-d', '--dataset', default='voc', help='voc or coco')
    parser.add_argument('--root', default='/home/zeyi/PycharmProjects/yolo_learning/data/dataset',
                        help='data root')  # root指定到VOCdevkit所在目录

    return parser.parse_args()


def train():
    # 创建命令行参数
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # 保存模型的路径
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)  # exist_ok为True，则在目标目录已经存在的情况下不会触发FileExistsError异常

    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        device = torch.device('cuda')
    else:
        print('use cpu')
        device = torch.device('cpu')

    # 是否使用多尺度训练技巧
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416

    # 构建dataset类
    dataset, num_classes, evaluator = build_dataset(args, device, train_size, val_size)

    # 构建dataloader类
    dataloader = build_dataloader(args, dataset)

    # 构建模型
    model = build_yolo(args, device, train_size, num_classes, trainable=True)
    model.to(device).train()

    # 计算模型的FLOPs和参数量
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    model_copy.set_grid(val_size)
    FLOPs_and_Params(model_copy, val_size, device)
    del model_copy

    # 使用 tensorboard 可视化训练过程，是否使用tensorboard来保存训练过程中的各类数据
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log_path = os.path.join('log', args.dataset, args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # 构建优化器
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    max_epoch = args.max_epoch  # 最大训练轮次
    lr_epoch = args.lr_epoch  # 学习率衰退的轮次，默认为[90, 120]
    epoch_size = len(dataloader)  # 每一训练轮次的迭代次数

    # 开始训练
    best_map = -1.  # 记录最好的mAP
    t0 = time.time()
    for epoch in range(args.start_epoch, max_epoch):

        # 使用阶梯学习率衰减策略
        if epoch in lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size  # 当前迭代次数（batch处理次数，每次迭代处理batch_size个数据），不是迭代轮次
            # 使用warm-up策略来调整早期的学习率
            # warmup的作用在于可以缓解模型在训练初期由于尚未学到足够好的参数而回传不稳定的梯度所导致的负面影响
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    nw = args.wp_epoch * epoch_size  # warmup阶段所需总的迭代次数
                    tmp_lr = base_lr * pow(ni * 1. / nw, 4)
                    set_lr(optimizer, tmp_lr)
                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)

            # 多尺度训练。每训练10次就随机抽取一个新的尺寸，用做接下来训练中的图像尺寸
            if args.multi_scale and iter_i % 10 == 0 and iter_i > 0:
                # 随机选择一个新的训练尺寸
                train_size = random.randint(10, 19) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # 插值
                images = F.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = gt_creator(input_size=train_size, stride=model.stride, label_lists=targets)

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # 正向传播和计算损失
            conf_loss, cls_loss, bbox_loss, total_loss = model(images, targets)

            # 梯度累加 & 反向传播
            total_loss /= args.accumulate  # 损失标准化。由于我们的loss是在batch维度上做归一化的，所以为了保证梯度累加的等效性，我们需要对计算出来的总的损失total_loss做平均
            total_loss.backward()  # 反向传播，计算梯度

            # 梯度累加达到固定次数之后，更新参数，然后将梯度清零
            if ni % args.accumulate == 0:  # 还是 if (ni+1) % args.accumulate == 0 不过影响应该不大
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 梯度清零

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', conf_loss.item(), ni)
                    writer.add_scalar('cls loss', cls_loss.item(), ni)
                    writer.add_scalar('box loss', bbox_loss.item(), ni)

                t1 = time.time()
                fmt_str = '[Epoch %d/%d][Iter %d/%d][lr %.6f]' \
                          '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time %.2f]' % (
                              epoch + 1, max_epoch, iter_i, epoch_size, tmp_lr, conf_loss.item(), cls_loss.item(),
                              bbox_loss.item(), total_loss.item(), train_size, t1 - t0
                          )
                print(fmt_str, flush=True)
                t0 = time.time()

        # 验证
        if epoch % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate()


def set_lr(optimizer: Optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_dataset(args, device, train_size, val_size):
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)  # BGR
    train_transform = Augmentation(train_size, pixel_mean, pixel_std)
    val_transform = BaseTransform(val_size, pixel_mean, pixel_std)

    # 构建dataset
    if args.dataset == 'voc':
        data_root = os.path.join(args.root, 'VOCdevkit')
        # 加载voc数据集
        num_classes = 20
        dataset = VOCDetection(data_root, image_sets=[('2012', 'train')], transform=train_transform)
        evaluator = VOCAPIEvaluator(data_root=data_root, img_size=val_size, device=device, transform=val_transform,
                                    year='2012')
    else:
        print('unknown dataset!! Only support vod!!!')
        exit(0)

    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, num_classes, evaluator


def build_dataloader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    train()
