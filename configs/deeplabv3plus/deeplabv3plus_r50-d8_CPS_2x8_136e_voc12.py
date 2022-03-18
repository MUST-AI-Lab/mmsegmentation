_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_136e.py'
]

model = dict(
    decode_head=dict(num_classes=20))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 300)
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ExtraAttrs', tag='sup'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=['filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'tag']),
]


unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Add_Pseudo_gt'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ExtraAttrs', tag='unsup'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=['filename', 'ori_filename', 'ori_shape',
                                                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                                    'flip_direction', 'img_norm_cfg', 'tag']),
]


"""
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='PascalVOCDataset',
        data_root='../data/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/train.txt',
        pipeline=sup_pipeline),
)
"""

data = dict(
    samples_per_gpu=4,
    train=dict(
        _delete_=True,
        type='SemiDataset',
        sup=dict(
            type='PascalVOCDataset',
            data_root='data/VOCdevkit/VOC2012',
            img_dir='JPEGImages',
            ann_dir='SegmentationClass',
            split='ImageSets/Segmentation/semi_partition/pseudoseg_labeled_1-8.txt',
            pipeline=sup_pipeline,
        ),
        unsup=dict(
            type='PascalVOCDataset',
            data_root='data/VOCdevkit/VOC2012',
            img_dir='JPEGImages',
            ann_dir='SegmentationClass',
            split='ImageSets/Segmentation/semi_partition/pseudoseg_unlabeled_1-8.txt',
            pipeline=unsup_pipeline,
        ),

    ),
    sampler=dict(
        train=dict(
            type='DistributedSemiSampler',
            sample_ratio=[1, 1],
        )
    )
)


semi_wrapper = dict(
    type="CrossPesudoSupervision",
    train_cfg=dict(unsup_weight=1.5),
)
"""
# Two optimizer for each branch
optimizer = dict(
    _delete_=True,
    branch1=dict(type='SGD', lr=0.00031, momentum=0.9, weight_decay=0.0005),
    branch2=dict(type='SGD', lr=0.00031, momentum=0.9, weight_decay=0.0005),
)
"""
optimizer = dict(type='SGD', lr=0.00031, momentum=0.9, weight_decay=0.0005)