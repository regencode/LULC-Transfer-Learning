optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00005, weight_decay=0.01),
    dtype="bfloat16",
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="LinearLR", start_factor=1.0, end_factor=0.0, by_epoch=True, begin=0, end=100),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=10,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=False),
)

custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="mIoU",
        rule="greater",
        patience=10,
        min_delta=0.0001,
    ),
]
