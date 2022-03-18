_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/camvid_video.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(num_classes=12),
    auxiliary_head=dict(num_classes=12),
    test_cfg = dict(mode='whole'),
)
evaluation = dict(metric=['mDice','mIoU', 'mFscore'])

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0005)