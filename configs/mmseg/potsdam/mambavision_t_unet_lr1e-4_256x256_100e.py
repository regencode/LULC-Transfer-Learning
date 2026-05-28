_base_ = [
    "../_base_/models/mambavision_t_unet.py",
    "../_base_/datasets/potsdam_256x256.py",
    "../_base_/schedules/schedule_adamw_100e.py",
    "../_base_/default_runtime.py",
]


model = dict(
    backbone=dict(
        pretrained="/tmp/mamba_vision_T.pth.tar",
    ),
)
