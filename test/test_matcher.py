import os
import sys
import unittest
from typing import List
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from retinanet.model.utils import *
from retinanet.model.box_utils import matcher


class TestMatcher(unittest.TestCase):
    def test_scriptability(self):
        # Sample anchors and targets tensors (assuming XYXY format)
        anchors = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]])
        targets = torch.tensor([[15, 15, 45, 45], [25, 25, 65, 65]])

        matches = matcher(anchors, targets)

        expected_matches = torch.tensor([0, 1, 1])
        
        self.assertTrue(torch.allclose(matches, expected_matches))

if __name__ == "__main__":
    unittest.main()