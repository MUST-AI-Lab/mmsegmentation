_base_ = [
    '../_base_/models/fcn_unet_s5-d16_noAux_dist.py', '../_base_/datasets/drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    test_cfg=dict(crop_size=(64, 64), stride=(42, 42))
)
evaluation = dict(metric=['mDice','mIoU', 'mFscore'])


semi_wrapper = dict(
    type="DML_triple",
    train_cfg=dict(unsup_weight=1, t=2),
    test_cfg=dict(inference_on='branch1'),
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)



