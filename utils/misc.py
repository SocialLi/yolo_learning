import torch


def detection_collate(batch):
    """
    为dataloader类写一个collection_fn函数。
    在训练中，我们通常使用多张图像组成一批数据去训练，即所谓的mini-batch概念。

    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images, dim=0), targets
