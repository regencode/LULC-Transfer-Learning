_base_ = ["../_base_/models/mambavision_t_upernet.py"]

default_scope = "mmseg"

custom_imports = dict(
    imports=[
        "transferlearning.datasets.mmseg_isprs",
        "transferlearning.models.backbones.mmseg_mambavision",
    ],
    allow_failed_imports=False,
)

dataset_type = "ISPRSPotsdamDataset"
data_root = "data/potsdam_patch256"

pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images", seg_map_path="ann_dir"),
        ann_file="splits_overfit/train.txt",
        pipeline=pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images", seg_map_path="ann_dir"),
        ann_file="splits_overfit/val.txt",
        pipeline=pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.01),
)

param_scheduler = [
    dict(type="ConstantLR", factor=1.0),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=20, val_interval=10)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=20),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

log_processor = dict(by_epoch=False)
