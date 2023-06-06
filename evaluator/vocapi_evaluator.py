import os.path
import time

import torch

from data.voc0712 import VOC_CLASSES, VOCDetection
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
from xml.etree import ElementTree as ET
from numpy.typing import NDArray
from typing import List


class VOCAPIEvaluator:
    """ VOC AP Evaluation class """

    def __init__(self, data_root, img_size, device, transform, set_type='test', year='2007', display=False):
        self.data_root = data_root
        self.img_size = img_size
        self.device = device
        self.transform = transform
        self.labelmap = VOC_CLASSES
        self.set_type = set_type
        self.year = year
        self.display = display

        # path
        folder = 'VOC' + year
        self.devkit_path = os.path.join(data_root, folder)
        self.annopath = os.path.join(self.devkit_path, 'Annotations', '%s.xml')
        self.imgpath = os.path.join(self.devkit_path, 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(self.devkit_path, 'ImageSets', 'Main', set_type + '.xml')
        self.output_dir = self.get_output_dir('voc_eval/', self.set_type)

        # dataset
        self.dataset = VOCDetection(root=data_root, image_sets=[(year, set_type)], transform=transform)

        self.all_boxes = []
        self.map = None


    def evaluate(self, net: nn.Module):
        net.eval()
        num_images = len(self.dataset)
        # all detections are collected into:
        #   all_boxes[cls][image] = N x 5 array of detections in
        #   (x1, y1, x2, y2, score)
        self.all_boxes = [[[] for _ in range(num_images)] for _ in range(len(self.labelmap))]

        # timers
        det_file = os.path.join(self.output_dir, 'detections.pkl')

        # 遍历当前数据集下所有图片
        for i in range(num_images):
            im, gt, h, w = self.dataset.pull_item(i)
            x = Variable(im.unsqueeze(0)).to(self.device)  # [1,C,H,W]
            t0 = time.time()
            # forward
            bboxes, scores, labels = net(x)  # 返回类型 Tensor,Tensor,Tensor
            detect_time = time.time() - t0

            # 将归一化的bboxes还原为当前图像下的实际大小
            scale = torch.tensor([w, h, w, h])
            bboxes *= scale

            # 遍历所有类别
            for j in range(len(self.labelmap)):
                inds = torch.nonzero(labels == j).flatten()
                if inds.numel() == 0:  # 当前预测该类别的bbox个数为0
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                self.all_boxes[j][i] = c_dets

            if i % 500 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        with open(det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self.evaluate_detections(self.all_boxes)
        print('Mean AP: ', self.map)

    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()

    def write_voc_results_file(self, all_boxes):
        # type: (List[List[NDArray]]) -> None
        """
        将检测结果写入文件，每个类别写一个文件。写入内容的每行格式为: 图片名 score x1 y1 x2 y2
        :param all_boxes: 检测结果，第1个索引为类别，第2个索引为图片index(全部图片的第几个)
        :return: None
        """
        # 遍历类型
        for cls_ind, cls in enumerate(self.labelmap):
            if self.display:
                print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(cls)
            with open(filename, 'wt') as f:
                # 遍历数据集所有图片
                # 数据集保存的图片列表的索引和all_boxes的图片索引一一对应，因此遍历数据集就能拿到图片名，并根据索引拿到all_boxes存储的检测结果
                for im_ind, index in enumerate(self.dataset.ids):
                    # 这就是为什么all_boxes格式为 all_boxes[cls][image] = N x 5 array of detections in
                    # (x1, y1, x2, y2, score)
                    det_boxes = all_boxes[cls_ind][im_ind]
                    if det_boxes == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(det_boxes.shape[0]):
                        f.write('{:s} {:.3f} {:1f} {:.1f} {:.1f} {:.1f}\n'.format(index[1], det_boxes[k, -1],
                                                                                  det_boxes[k, 0] + 1,
                                                                                  det_boxes[k, 1] + 1,
                                                                                  det_boxes[k, 2] + 1,
                                                                                  det_boxes[k, 3] + 1))

    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        # 对每个类别计算AP
        for i, cls in enumerate(self.labelmap):
            filename = self.get_voc_results_file_template(cls)
            rec, prec, ap = self.voc_eval(detpath=filename, classname=cls, cachedir=cachedir, ovthresh=0.5,
                                          use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        self.map = np.mean(aps)
        if self.display:
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(self.map))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('--------------------------------------------------------------')
        else:
            print('Mean AP = {:.4f}'.format(np.mean(self.map)))

    def voc_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        """

        :param detpath: 当前类别的检测结果文件所在的路径
        :param classname: 当前类别名
        :param cachedir: 缓存目录，用pickle保存所有文件对应的object，即保存{文件名: object列表}到一个文件中，方便后续直接反序列化
        :param ovthresh: iou阈值。
        :param use_07_metric:
        :return: rec, prec, ap
        """
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')

        # 读取所有图片的标注信息，并序列化保存，如果序列化文件已经存在则直接读取
        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
        image_names = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}  # recs保存所有文件对应的object, {文件名: object列表}
            for i, image_name in enumerate(image_names):
                recs[image_name] = self.parse_rec(self.annopath % image_name)
                if i % 100 == 0 and self.display:
                    print('Reading annotation for {:d}/{:d}'.format(i + 1, len(image_names)))
            # save
            if self.display:
                print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # 根据上面加载的所有图片的标注信息，继续获取具体类别的标注信息
        # 也即遍历上面加载的recs，对于每个图片，获取类别为当前类别的标注信息，将它们的bbox，difficult等信息分别汇总到一起，并生成嵌套字典 {图片名：dict}
        # 生成的字典为class_recs [注] key包含每个图片名，即使该图片下没有该类型的标注信息
        class_recs = {}
        npos = 0  # 记录所有正样本个数（非difficult）
        for image_name in image_names:
            R = [obj for obj in recs[image_name] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

        # 读取本类别的检测结果，每行为：图片名 score x1 y1 x2 y2
        with open(detpath, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:  # 判断lines列表是否有非空字符串，即是否有检测结果
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # 根据置信度降序排序
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)  # 检测结果个数
            tp = np.zeros(nd)  # True Positive
            fp = np.zeros(nd)  # False Positive
            # 遍历该类别下的检测结果，统计TP和FP
            for d in range(nd):
                # [注] class_recs 保存的是真实标注信息，key就是图片名。
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)  # 预测 bbox，个数为1
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)  # 真实 bbox，个数可能为0-n
                if BBGT.size > 0:  # 如果图片中有该类别的目标框
                    # 计算 iou
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) + (BBGT[:, 2] - BBGT[:, 0]) * (
                            BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    # 获取iou最大值
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:  # 防止两个框预测同一个目标，则第二个框以及后续的框判定为分类错误
                            fp[d] = 1.
                else:
                    # 与真实样本的最大iou低于阈值，则分类错误
                    fp[d] = 1.

            # 利用confidence给BBOX排序并统计TP和FP之后，对每个BBOX都计算对应的precision和recall值
            # 可参考链接 https://zhuanlan.zhihu.com/p/67279824
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            # 计算recall
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # 计算precision
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # np.finfo(np.float64).eps用于防止除0
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            # 检测结果为空
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap

    def voc_ap(self, rec, prec, use_07_metric=True):
        """计算AP
        参考链接 https://zhuanlan.zhihu.com/p/67279824
        Compute VOC AP given precision and recall. If use_07_metric is true, uses the VOC 07 11 point method (default:True).
        :param rec: recall
        :param prec: precision
        :param use_07_metric:
        :return:
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0.
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # 正确的 AP 计算
            # 首先在末尾附加哨兵值
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def parse_rec(self, filename):
        # type: (str) -> List[dict]
        """Parse a PASCAL VOC xml file"""
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)
        return objects

    def get_output_dir(self, name, phase):
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir

    def get_voc_results_file_template(self, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + self.set_type + '_%s.txt' % cls
        filedir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path
