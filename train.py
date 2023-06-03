import argparse
import os
import torch
from data.transform import BaseTransform


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
    parser.add_argument('-accu', '--accumulate', default=8, type=int, help='gradient accumulate.')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1, help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, help='keep training')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False, help='use multi-scale trick')
    parser.add_argument('--max_epoch', type=int, default=150, help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[90, 120], type=int, help='lr epoch to decay')

    # 优化器参数
    parser.add_argument('-lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

    # 数据集参数
    parser.add_argument('-d', '--dataset', default='voc', help='voc or coco')
    parser.add_argument('--root', default='D:/code/PycharmProjects/yolo_learning/data/dataset',
                        help='data root')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)  # exist_ok为True，则在目标目录已经存在的情况下不会触发FileExistsError异常

    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 是否使用多尺度训练
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416

    # 构建dataset类
    build_dataset(args, device, train_size, val_size)


def build_dataset(args, device, train_size, val_size):
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)  # BGR
    # train_transform = Augu
    val_transform = BaseTransform(val_size, pixel_mean, pixel_std)


def build_dataloader(args, dataset):
    pass


if __name__ == '__main__':
    train()
