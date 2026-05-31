dataset_type = "ISPRSPotsdamDataset"
data_root = "data/potsdam_patch256"

img_suffix = ".tif"
seg_map_suffix = ".png"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs"),
]

custom_imports = dict(
    imports=[
        "transferlearning.datasets.mmseg_isprs",
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

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
test_evaluator = val_evaluator
