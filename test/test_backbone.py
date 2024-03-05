import os
import sys
import unittest
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from retinanet.model.utils import *
from retinanet.model.backbone import get_backbone
from retinanet.model.layers import *

__small__ = ["resnet18", "resnet34"]
__big__ = ["resnet50", "resnet101", "resnet101", "resnet152"]


class TestBackBone(unittest.TestCase):

    def test_resnet_scriptability(self):
        resnet = get_backbone(BACKBONE, PRETRAINED_BACKBONE, freeze_bn=FREEZE_BN)
        if BACKBONE in __small__:
            fpn_szs = [
                resnet.backbone.layer2[1].conv2.out_channels,
                resnet.backbone.layer3[1].conv2.out_channels,
                resnet.backbone.layer4[1].conv2.out_channels,
            ]
        elif BACKBONE in __big__:
            fpn_szs = [
                resnet.backbone.layer2[2].conv3.out_channels,
                resnet.backbone.layer3[2].conv3.out_channels,
                resnet.backbone.layer4[2].conv3.out_channels,
            ]
        
        inp = torch.rand(2, 3, 112, 112)
        
        out = resnet(inp)

        self.assertTrue(out[0].shape[1] == fpn_szs[0])
        self.assertTrue(out[1].shape[1] == fpn_szs[1])
        self.assertTrue(out[2].shape[1] == fpn_szs[2])
    

if __name__ == "__main__":
    unittest.main()

        
