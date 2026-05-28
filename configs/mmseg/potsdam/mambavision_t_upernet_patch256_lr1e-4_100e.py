_base_ = [
    "../_base_/models/mambavision_t_upernet.py",
    "../_base_/datasets/potsdam_patch256.py",
    "../_base_/schedules/schedule_adamw_100e.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(
    imports=[
        "transferlearning.datasets.mmseg_isprs",
        "transferlearning.models.backbones.mmseg_mambavision",
    ],
    allow_failed_imports=False,
)
