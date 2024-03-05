import os
import sys
import unittest
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from retinanet.model.utils import *
from retinanet.model.backbone import get_backbone
from retinanet.model.layers import FeaturePyramid, RetinaNetHead, RetinaNetBoxSubnet, RetinaNetClassSubnet , RetinaNetLosses

__small__ = ["resnet18", "resnet34"]
__big__ = ["resnet50", "resnet101", "resnet101", "resnet152"]


class TestBackBone(unittest.TestCase):

    def test_fpn(self):
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

        fpn = FeaturePyramid(fpn_szs[0], fpn_szs[1], fpn_szs[2], 256)
        
        retinanet_head = RetinaNetHead(256, 256, 2220, NUM_CLASSES, PRIOR)

        inp = torch.rand(2, 3, 224, 224)
        
        out = resnet(inp)

        feature_maps = fpn(out)

        out = retinanet_head(feature_maps)

        print(out)

if __name__ == "__main__":
    unittest.main()

        
