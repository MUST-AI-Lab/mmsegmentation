_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/datas6.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    decode_head=dict(num_classes=23),
    auxiliary_head=dict(num_classes=23),
    test_cfg = dict(mode='slide', crop_size=(256, 256), stride=(85, 85))
)