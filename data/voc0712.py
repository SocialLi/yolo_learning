import os

import torch
import torch.utils.data as data
import os.path as osp
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from torch import Tensor
from typing import List, Tuple
from numpy.typing import NDArray

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height) -> List[List]:
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

     Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, img_size=None, image_sets=[('2007', 'trainval'), ('2012', 'trainval')], transform=None,
                 target_transform=VOCAnnotationTransform(), dataset_name='VOC0712'):
        self.root = root  # root指定到`VOC2012`所在的目录
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform  # 这个self.transform变量就是用来预处理VOC数据的，处理好的数据即可用于网络训练或测试。
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            if not os.path.exists(rootpath):
                continue
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        # type: (int) -> Tuple[Tensor, NDArray, int, int]
        """
        一共会返回四个变量，分别是图像(CHW,有transform则RGB)、标注数据、原始图像的高和原始图像的宽，所谓的原始图像，即 未经过预处理的数据集图像。
        :param index:
        :return:
        """
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

        # Augmentation or BaseTransform
        """
        预处理一共包括两种方式：数据增强（data augmentation）和基础变换（base transform）。
        - 数据增强是使用一系列的诸如随机裁剪、随机翻转、随机图像色彩变换等手段对图像进行处理;
        - 基础变换则是普通的图像归一化操作，即将所有像素除以255，再使用ImageNet上的均值和方差做归一化。
        这个self.transform变量就是用来预处理VOC数据的，处理好的数据即可用于网络训练或测试。
        """
        if self.transform is not None:
            # check labels
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
            # 经过变换boxes坐标可能发生变换，同时boxes和labels数量也可能减少（如裁剪），因此才需要传入 boxes和labels
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # 一共会返回四个变量，分别是图像、标注数据、原始图像的高和原始图像的宽，所谓的原始图像，即 未经过预处理的数据集图像。
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        # type: (int) -> Tuple[NDArray, Tuple]
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id

    def pull_anno(self, index):
        # type: (int) -> Tuple[str, List[List]]
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, width=1, height=1)  # w,h传入1,1使归一化采用的高宽为1,1,这样就不会改变原始锚框的坐标
        return img_id[1], gt


if __name__ == '__main__':
    from transform import BaseTransform, Augmentation

    img_size = 640
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)  # BGR
    data_root = '/home/zeyi/PycharmProjects/yolo_learning/data/dataset/VOCdevkit'
    train_transform = Augmentation(img_size, pixel_mean, pixel_std)
    val_transform = BaseTransform(img_size, pixel_mean, pixel_std)

    dataset = VOCDetection(
        root=data_root,
        img_size=img_size,
        image_sets=[('2007', 'trainval')],
        transform=train_transform
    )

    for i in range(10):
        # dataset返回增强后图像的维度顺序是CHW,同时通道顺序为RGB
        im, gt, h, w = dataset.pull_item(i)

        # CHW -> HWC
        image: NDArray = im.permute(1, 2, 0).numpy()
        # RGB -> BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = (image * pixel_std + pixel_mean) * 255
        # to uint8
        image = image.astype(np.uint8).copy()

        # draw bbox
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            image = cv2.rectangle(image, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(0, 0, 255),
                                  thickness=2)

        cv2.imshow('gt', image)
        cv2.waitKey(0)
