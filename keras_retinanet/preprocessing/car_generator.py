"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path


def _process_dataset_(root):
    train_set = set(open(root + '/ImageSets/train.txt').read().splitlines())
    images = [
        (root + '/Images/' + img, open(root + '/Annotations/' + os.path.splitext(img)[0] + '.txt').read().splitlines())
        for img in os.listdir(root + '/Images') if os.path.splitext(img)[0] in train_set]
    return images


class CarsGenerator(Generator):
    """ Generate data for cars datasets.
    """

    def __init__(
            self,
            base_dir='../',
            **kwargs
    ):
        """ Initialize a cars data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched.
        """

        images = _process_dataset_(base_dir + '/datasets/PUCPR+_devkit/data')
        images += _process_dataset_(base_dir + '/datasets/CARPK_devkit/data')
        images = {k: [[int(n) for n in s.split()] for s in v] for k, v in images}

        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir

        self.labels = {1: '1'}
        self.classes = {'1': 1}

        self.image_names = list(images.keys())
        self.image_data = {img: [{'x1':box[0], 'y1': box[1], 'x2':box[2], 'y2':box[3], 'class': self.labels[box[4]]}
                                 for box in data] for img, data in images.items()}
        super(CarsGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return self.image_names[image_index]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations
