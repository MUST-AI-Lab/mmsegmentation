_base_ = [
    '../_base_/models/fcn_unet_s5-d16_noAux.py', '../_base_/datasets/tray_food_smallCrop.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=42),
    test_cfg=dict(mode='whole')
)
evaluation = dict(metric=['mDice','mIoU', 'mFscore'])


semi_wrapper = dict(
    type="DML",
    train_cfg=dict(unsup_weight=0),
    test_cfg=dict(inference_on='branch1'),
)

custom_hooks = [
    dict(type='StepLossWeightUpdateHook', step=500, gamma=2, max_weight=30, priority='NORMAL')
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)



