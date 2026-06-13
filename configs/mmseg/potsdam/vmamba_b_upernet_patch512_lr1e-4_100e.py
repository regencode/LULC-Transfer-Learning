_base_ = [
    "../_base_/models/vmamba_b_upernet.py",
    "../_base_/datasets/potsdam_patch512.py",
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
    backbone=dict(pretrained="/tmp/vssm1_base_0229s_ckpt_epoch_260.pth"),
)
