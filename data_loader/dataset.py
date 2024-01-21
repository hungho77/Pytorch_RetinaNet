from __future__ import print_function, division
import sys
import os
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
import random
import csv

from torch.utils.data.sampler import Sampler
from PIL import Image

from base import BaseDataset


class CocoDataset(BaseDataset):
    """Coco dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self, 
            root: str, 
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
        ) -> None:
        super(CocoDataset, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = (list(sorted(self.coco.imgs.keys())))

        self.load_classes()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple = Tuple (Image, target). target is the object returned by load annotations from labels.
        """ 

        img = self.load_image(index)
        label = self.load_labels(index)

        if self.transforms is not None:
            img, label = self.transforms(img, label)
        return img, label

    def __len__(self):
        return len(self.ids)
    
    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key


    def load_image(self, index):
        """Load image from file with index.

        Args:
            index (int): Index

        Returns:
            PIL Image: Image
        """
        img_id = self.ids[index]
        image_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, image_info['file_name'])

        img = Image.open(path).convert('RGB')

        return img

    def load_labels(self, index):
        """Load labels with index.

        Args:
            index (int): Index
        
        Returns:
            str: Target
        """

        # Get groundt truth coco annotations ids
        ann_ids = self.coco.getAnnIds(imgIds=self.ids[index], iscrowd=False)
        # Create an empty labels with shape (0, 5)
        labels = np.zeros((0, 5))  
        
        # Missing labels
        if len(ann_ids) == 0:
            return labels

        # Parse annotations
        coco_annotations = self.coco.loadAnns(ann_ids)
        for a in coco_annotations:
            # Missing width or height
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            label = np.zeros((1, 5))
            label[0, :4] = a['bbox']
            label[0, 4] = self.coco_label_to_label(a['category_id'])
            labels = np.append(labels, label, axis=0)
        
        # Tranform from [x, y, w, h] to [x1, y1, x2, y2]
        labels[:, 2] = labels[:, 0] + labels[:, 2]
        labels[:, 3] = labels[:, 1] + labels[:, 3]

        return labels

    def coco_label_to_label(self, coco_label):
        """Convert coco label format to label format

        Args:
            coco_label (dict): coco label format

        Returns:
            dict: label format
        """
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        """Convert label fromat to coco label format

        Args:
            label (dict): label format

        Returns:
            dict: coco label format
        """
        return self.coco_labels[label]
    
