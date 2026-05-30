_base_ = [
    "../_base_/models/vmamba_s_upernet.py",
    "../_base_/datasets/potsdam_patch256.py",
    "../_base_/schedules/schedule_adamw_100e.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(
    imports=[
        "transferlearning.datasets.mmseg_isprs",
        "transferlearning.models.backbones.mmseg_vmamba",
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(pretrained="/tmp/vssm1_small_0230s_ckpt_epoch_222.pth"),
)
