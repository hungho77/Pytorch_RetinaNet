import os
import sys
import logging
import unittest
import random

from PIL import Image
import torch
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms import functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from retinanet.model.anchors import AnchorGenerator
from retinanet.model.utils import *

logger = logging.getLogger(__name__)


class TestAnchorGenerator(unittest.TestCase):
    def test_default_anchor_generator(self):
        anchor_generator = AnchorGenerator()

        # only the last two dimensions of features matter here
        num_images = 2
        # Assuming you have a list of PIL Images
        image_list = [Image.new('RGB', (224, 224)) for _ in range(num_images)]

        # Randomly shuffle the list of images
        random.shuffle(image_list)

        # Convert PIL Images to PyTorch tensors
        image_tensors = [F.to_tensor(img) for img in image_list]

        # Create an ImageList instance
        image_list_instance = ImageList(image_tensors, image_sizes=[img.shape[-2:] for img in image_tensors])
        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        anchors = anchor_generator(image_list_instance, [features["stage3"]])
        expected_anchor_tensor = torch.tensor(
            [
                [-22.6274, -11.3137,  22.6274,  11.3137],
                [-16.0000, -16.0000,  16.0000,  16.0000],
                [-11.3137, -22.6274,  11.3137,  22.6274],
                [-28.5088, -14.2544,  28.5088,  14.2544],
                [-20.1587, -20.1587,  20.1587,  20.1587],
                [-14.2544, -28.5088,  14.2544,  28.5088],
                [-35.9188, -17.9594,  35.9188,  17.9594],
                [-25.3984, -25.3984,  25.3984,  25.3984],
                [-17.9594, -35.9188,  17.9594,  35.9188],
                [-14.6274, -11.3137,  30.6274,  11.3137],
                [ -8.0000, -16.0000,  24.0000,  16.0000],
                [ -3.3137, -22.6274,  19.3137,  22.6274],
                [-20.5088, -14.2544,  36.5088,  14.2544],
                [-12.1587, -20.1587,  28.1587,  20.1587],
                [ -6.2544, -28.5088,  22.2544,  28.5088],
                [-27.9188, -17.9594,  43.9188,  17.9594],
                [-17.3984, -25.3984,  33.3984,  25.3984],
                [ -9.9594, -35.9188,  25.9594,  35.9188]
            ]
        )

        self.assertTrue(torch.allclose(anchors[0], expected_anchor_tensor))

    def test_default_anchor_generator_centered(self):
        # test explicit args
        anchor_generator = AnchorGenerator(
            sizes=[32, 64], aspect_ratios=[0.25, 1, 4], strides=[4]
        )

        # only the last two dimensions of features matter here
        num_images = 2
        # Assuming you have a list of PIL Images
        image_list = [Image.new('RGB', (224, 224)) for _ in range(num_images)]

        # Randomly shuffle the list of images
        random.shuffle(image_list)

        # Convert PIL Images to PyTorch tensors
        image_tensors = [F.to_tensor(img) for img in image_list]

        # Create an ImageList instance
        image_list_instance = ImageList(image_tensors, image_sizes=[img.shape[-2:] for img in image_tensors])

        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        expected_anchor_tensor = torch.tensor(
            [[-32.,  -8.,  32.,   8.],
            [-16., -16.,  16.,  16.],
            [ -8., -32.,   8.,  32.],
            [-64., -16.,  64.,  16.],
            [-32., -32.,  32.,  32.],
            [-16., -64.,  16.,  64.],
            [-28.,  -8.,  36.,   8.],
            [-12., -16.,  20.,  16.],
            [ -4., -32.,  12.,  32.],
            [-60., -16.,  68.,  16.],
            [-28., -32.,  36.,  32.],
            [-12., -64.,  20.,  64.]]
        )

        anchors = anchor_generator(image_list_instance, [features["stage3"]])
        self.assertTrue(torch.allclose(anchors[0], expected_anchor_tensor))


if __name__ == "__main__":
    unittest.main()