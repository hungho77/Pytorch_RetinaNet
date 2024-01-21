import os
import sys

import torch
from torchvision.transforms import v2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data_loader.dataset import CocoDataset



def test_coco_dataset():
    # Init parameters

    root = "./data/small_coco/train2017"  # image dir path
    annFile = "./data/small_coco/annotations/instances_train2017.json"  # label file path
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    coco_dataset = CocoDataset(root=root, annFile=annFile, transform=transform)
    print(coco_dataset[0])


if __name__ == "__main__":
    test_coco_dataset()