_base_ = [
    "../_base_/models/resnet50_unet.py",
    "../_base_/datasets/potsdam_256x256.py",
    "../_base_/schedules/schedule_adamw_100e.py",
    "../_base_/default_runtime.py",
]

