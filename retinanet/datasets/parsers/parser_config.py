from dataclasses import dataclass

__all__ = ['CocoParserCfg']


@dataclass
class CocoParserCfg:
    ann_filename: str  # absolute path
    include_masks: bool = False
    include_bboxes_ignore: bool = False
    has_labels: bool = True
    bbox_yxyx: bool = True
    min_img_size: int = 32
    ignore_empty_gt: bool = False
