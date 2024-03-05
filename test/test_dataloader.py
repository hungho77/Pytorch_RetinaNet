import os
import sys

import torch
from torchvision.transforms import v2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from retinanet import create_loader, create_dataset
from retinanet.datasets import resolve_input_config, SkipSubset



if __name__ == "__main__":
    
    # input_config = resolve_input_config(args, model_config=model_config)

    dataset_train, dataset_eval = create_dataset('coco2017', '/workspace/hunght25/Pytorch_RetinaNet/sample/small_coco')

    # # setup labeler in loader/collate_fn if not enabled in the model bench
    # labeler = None
    # # if not args.bench_labeler:
    # #     labeler = AnchorLabeler(
    # #         Anchors.from_config(model_config),
    # #         model_config.num_classes,
    # #         match_threshold=0.5,
    # #     )
    
    loader_train = create_loader(
        dataset_train,
        input_size=224,
        batch_size=16,
        is_training=True,
        use_prefetcher=True,
        re_prob=0,
        re_mode='pixel',
        re_count=1,
        # color_jitter=args.color_jitter,
        # auto_augment=args.aa,
        interpolation='bilinear',
        fill_color='mean',
        # mean=input_config['mean'],
        # std=input_config['std'],
        # num_workers=args.workers,
        distributed=False,
        pin_mem=False,
        anchor_labeler=None,
        transform_fn=None,
        collate_fn=None,
    )

    if 2 > 1:
        dataset_eval = SkipSubset(dataset_eval, 2)
    loader_eval = create_loader(
        dataset_eval,
        input_size=224,
        batch_size=16,
        is_training=False,
        use_prefetcher=True,
        interpolation='bilinear',
        fill_color='mean',
        # mean=input_config['mean'],
        # std=input_config['std'],
        # num_workers=args.workers,
        distributed=False,
        pin_mem=False,
        anchor_labeler=None,
        transform_fn=None,
        collate_fn=None,
    )

    for batch_idx, (input, target) in enumerate(loader_train):
        print(batch_idx)
        print(input)
        print(target)

    for batch_idx, (input, target) in enumerate(loader_eval):
        print(batch_idx)
        print(input)
        print(target)