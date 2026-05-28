dataset_type = "ISPRSPotsdamDataset"
data_root = "data/potsdam"

img_suffix = ".tif"
seg_map_suffix = ".png"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="Resize", scale=(256, 256), keep_ratio=False),
    dict(type="SyncOriShape"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PhotoMetricDistortion",
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="Resize", scale=(256, 256), keep_ratio=False),
    dict(type="SyncOriShape"),
    dict(type="PackSegInputs"),
]

custom_imports = dict(
    imports=[
        "transferlearning.datasets.mmseg_isprs",
        "transferlearning.datasets.transforms",
    ],
    allow_failed_imports=False,
)

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="images",
            seg_map_path="ann_dir",
        ),
        ann_file="splits/train.txt",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="images",
            seg_map_path="ann_dir",
        ),
        ann_file="splits/val.txt",
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
