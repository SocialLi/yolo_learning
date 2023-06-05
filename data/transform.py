import cv2
import numpy as np
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union


class Compose:
    """Composes several augmentations together.
       Args:
           transforms (List[Transform]): list of transforms to compose.
       Example:
           >>> augmentations.Compose([
           >>>     transforms.CenterCrop(10),
           >>>     transforms.ToTensor(),
           >>> ])
       """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts:
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels


class ToAbsoluteCoords:
    """
    将归一化的相对坐标转换为绝对坐标
    """

    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords:
    """
    将绝对坐标转化为归一化的相对坐标
    """

    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ConvertColor:
    """
    转换图像的色彩空间
    """

    def __init__(self, current='BGR', transform='HSV'):
        self.current = current
        self.transform = transform

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class Resize:
    def __init__(self, size=640):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomHue:
    """
    图像色调变化同饱和度，在HSV空间内对色调这一维的值进行加减。
    """

    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            # 规范超过范围的像素值
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomSaturation:
    """
    图像饱和度是指色彩纯度，纯度越高，则看起来更加鲜艳；纯度越低，则看起来较黯淡。如我们常说的红色比淡红色更加“红”，就是说红色的饱和度比淡红色的饱和度更大。

    数据增强中的随机饱和度的思想是在HSV空间内对饱和度这一维的值进行缩放。
    所以，我们首先需要将图像从RGB空间转换到HSV空间。同时，我们将其乘上一个随机因子，
        - 当该因子的值小于1时，图像的饱和度会减小；
        - 当该因子的值大于1时，图像的饱和度会变大。
    """

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            # 随机缩放S空间的值
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomLightNoise:
    """
    在RGB空间内随机交换通道的值，这样不同值的叠加最后也会得到不同的值
    """

    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)

        return image, boxes, labels


class RandomContrast:
    """
    图像对比度的定义是一幅图像中明暗区域最亮的白和最暗的黑之间不同亮度层级的测量，视觉上就是整幅图像的反差。
    数据增强中的随机对比度的思想是给图像中的每个像素值乘以一个随机因子值，
        - 当该因子的值小于1时，图像整体的对比度会减小；
        - 当该因子的值大于1时，图像整体的对比度会增大。
    """

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            # 生成随机因子
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness:
    """
    将RGB空间内的像素值均加上或减去一个值就可以改变图像整体的亮度
    """

    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            # 图像中的每个像素加上一个随机值
            image += delta
        return image, boxes, labels


class RandomSampleCrop:
    """Crop
    随机裁剪旨在裁掉原图中的一部分，然后检查边界框或目标整体是否被裁掉。如果目标整体被裁掉，则舍弃这次随机过程。
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # 使用整个原始输入图像
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # 随机选择一种裁剪方式
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            # 随机到None直接返回
            if mode is None:
                return image, boxes, labels
            # 最小IoU和最大IoU
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 迭代50次
            for _ in range(50):
                current_image = image

                # 宽和高随机采样
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # 宽高比例不当
                if h / w < 0.5 or h / w > 2:
                    continue

                # 随机选取左上角
                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # 框坐标 x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # 求iou
                overlap = jaccard_numpy(boxes, rect)

                # 是否满足最小和最大重叠约束
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 裁剪图像
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # 框中心点坐标
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # m1和m2用于过滤符合条件的boxes： 中心点在裁剪框范围内的boxes
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                # 若mask值全为False，则没有符合条件边框，重新裁剪
                if not mask.any():
                    continue

                # 过滤符合条件的boxes和labels
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]

                # 根据图像变换调整box，包括两步：
                # 1）裁剪box
                # 2）裁剪后的box的坐标还是相对于原始图像，需要计算相对于裁剪图像左上角的坐标（即所有坐标减去裁剪框的左上角坐标）
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand:
    """
    随机放大图像
    首先确定一个随机放大的尺度，生成放大后的图像并将值全部赋值为一个值mean,随后在放大图像内随机选区一块和原始图像相同大小的区域，将原始图像原封不动地复制过去。
    同时，依次将图像和边界框信息进行平移得到变换后的结果。
    """

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        # 随机放大尺度
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        # 确定放大后的图像的维度，并将所有像素赋上默认值
        expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        # 随机选取一块和原始图像相同大小区域，将原始图像复制过去
        expand_image[int(top):int(top) + height, int(left):int(left) + width] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror:
    """
    随机镜像
    随机镜像相当于将图像沿着竖轴中心翻转即垂直翻转（水平翻转类似）
    """

    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        if random.randint(2):
            # 图像翻转，image[:, ::-1]相当于第二个维度里的元素逆序：原始列顺序0,1,2, 改变后顺序2,1,0。但是行顺序不变，相当于图像左右翻转了。
            image = image[:, ::-1]
            boxes = boxes.copy()
            # 改变标注框。水平翻转只需要改变x1,x2坐标
            # x1,y1,x2,y2的变化：w-x2,y1,w-x1,y2
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes, labels


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, *args, image):
        """
       Args:
           image (Tensor): image tensor to be transformed
       Return:
           a tensor with channels swapped according to swap
       """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    """
    将上述提到的基于 "针对像素" 的数据增强方法封装到一个类中。
    针对图像像素的数据增强主要是改变原图像中像素的值，而不改变图像目标的形状和图像的大小。
    经过处理后，图像的饱和度、亮度、明度、颜色通道、颜色空间等会发生发生变化。这类变换不会改变原图中的标注信息，即边界框和类别。
    该类数据增强包括：
        - 图像对比度
        - 图像饱和度
        - 图像色调
        - 亮度
        - 随机交换通道
    """

    def __init__(self):
        self.pd = [
            RandomContrast(),  # 随机对比度
            ConvertColor(transform='HSV'),  # 转换色彩空间
            RandomSaturation(),  # 随机饱和度
            RandomHue(),  # 随机色调
            ConvertColor(current='HSV', transform='BGR'),  # 转换色彩空间
            RandomContrast()  # 随机对比度
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            # 先进行随机对比度变换
            distort = Compose(self.pd[:-1])
        else:
            # 最后再进行随机对比度变换
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels


class Augmentation:
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),  # 将int类型转换为float32类型
            ToAbsoluteCoords(),  # 将归一化的相对坐标转换为绝对坐标
            PhotometricDistort(),  # 图像颜色增强
            Expand(self.mean),  # 扩充增强
            RandomSampleCrop(),  # 随机裁剪
            RandomMirror(),  # 随机水平镜像
            ToPercentCoords(),  # 将绝对坐标转化为归一化的相对坐标
            Resize(self.size),  # resize操作
            Normalize(self.mean, self.std)  # 图像颜色归一化
        ])

    def __call__(self, image, boxes, labels):
        return self.augment(image, boxes, labels)


class BaseTransform:
    """基础变换

    1. 归一化
    通过除以255即可将所有的像素值映射到01范围内，然后再使用均值和标准差做进一步的归一化处理。

    均值和标准差是从ImageNet数据集中统计出来的，按照RGB通道的顺序，均值为[0.485, 0.456, 0.406]，标准差为[0.229, 0.224, 0.225]，是目前自然图像中很常用的均值方差数据。
    当然，从ImageNet统计出来的图像均值和方差并不适用于其他任何领域，比如医学图像领域，毕竟后者的图像可不是自然图像。
    在实际操作中，要注意一点，由opencv读取进来的图像，其颜色通道是按照BGR顺序排列的，而不是RGB，因此，读者会在项目代码中发现上面的均值和标准差的数值是反着排的。

    2. resize
    读取进来的图像大小通常都是不一样的，或者大小不合适，在训练过程中，为了将若干张图像拼接成一批batch数据，需要这些图像具有相同的尺寸，
    一种常用的方法便是将所有的图像都resize到同一尺寸
    [缺点] 改变了图像的长宽比，易导致图像中的物体发生畸变
    [另一种方法] 填充0像素，以避免畸变失真的问题
    """

    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size  # resize的大小
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        # resize
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        # normalize
        image /= 255
        image -= self.mean
        image /= self.std

        return image, boxes, labels
