import os.path

from data.voc0712 import VOC_CLASSES, VOCDetection


class VOCAPIEvaluator:
    """ VOC AP Evaluation class """

    def __init__(self, data_root, img_size, device, transform, set_type='test', year='2012', display=False):
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
        self.dataset = VOCDetection(root=data_root, image_sets=[('2012', set_type)], transform=transform)

    def evaluate(self, net):
        pass

    def get_output_dir(self, name, phase):
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir
