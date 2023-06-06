import torch
import torch.nn as nn


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


def load_weight(model, path_to_ckpt=None):
    # check
    if path_to_ckpt is None:
        print('no weight file ...')
        return model

    checkpoint_state_dict = torch.load(path_to_ckpt, map_location='cpu')
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model
