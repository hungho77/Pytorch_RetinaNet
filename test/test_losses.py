import os
import sys
import unittest
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from retinanet.model.utils import *
from retinanet.model.losses import RetinaNetLosses  # Replace 'your_module' with the actual module name


class TestRetinaNetLosses(unittest.TestCase):
    def setUp(self):
        # Initialize your RetinaNetLosses class with necessary parameters
        self.num_classes = 10  # Replace with the actual number of classes
        self.losses = RetinaNetLosses(self.num_classes)

    def test_smooth_l1_loss(self):
        # Create dummy input and target tensors
        input_tensor = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
        target_tensor = torch.tensor([1.0, 3.0, 5.0])

        # Calculate smooth L1 loss using the method from RetinaNetLosses
        loss = self.losses.smooth_l1_loss(input_tensor, target_tensor)

        print("smooth_l1_loss: ", loss)

    def test_focal_loss(self):
        # Create dummy input and target tensors
        clas_pred = torch.tensor([0.0, 1.0, 1.0], requires_grad=True)
        clas_tgt = torch.tensor([1.0, 1.0, 1.0])

        # Calculate focal loss using the method from RetinaNetLosses
        loss = self.losses.focal_loss(clas_pred, clas_tgt)

        print("focal_l1_loss: ", loss)

    # Add more tests for other components of the RetinaNetLosses class as needed

if __name__ == '__main__':
    unittest.main()
