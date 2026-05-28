_base_ = [
    "../_base_/models/vmamba_tiny_unet.py",
    "../_base_/datasets/potsdam_256x256.py",
    "../_base_/schedules/schedule_adamw_100e.py",
    "../_base_/default_runtime.py",
]


model = dict(
    backbone=dict(
        pretrained="/tmp/vssm1_tiny_0230s_ckpt_epoch_264.pth",
    ),
)
