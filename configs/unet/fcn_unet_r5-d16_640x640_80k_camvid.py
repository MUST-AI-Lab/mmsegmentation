_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/camvid_video.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    decode_head=dict(num_classes=12),
    auxiliary_head=dict(num_classes=12),
    test_cfg = dict(mode='slide', crop_size=(640, 640), stride=(512, 512))
)