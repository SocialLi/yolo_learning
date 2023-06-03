import cv2
import numpy as np


class Augmentation:
    def __int__(self):
        pass


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
