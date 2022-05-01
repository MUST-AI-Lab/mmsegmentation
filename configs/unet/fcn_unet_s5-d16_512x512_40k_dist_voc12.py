_base_ = [
    '../_base_/models/fcn_unet_s5-d16_noAux_dist.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21),
    test_cfg=dict(mode='slide', crop_size=(256,256), stride=(170,170))
)
evaluation = dict(metric=['mDice','mIoU', 'mFscore'])

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.005)



