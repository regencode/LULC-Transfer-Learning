norm_cfg = dict(type="SyncBN", requires_grad=True)

POTSDAM_CLASS_PROPORTIONS = [31.35, 24.81, 22.07, 15.31, 1.71, 4.75]
NUM_CLASSES = 6
POTSDAM_CLASS_WEIGHTS = [
    sum(POTSDAM_CLASS_PROPORTIONS) / (NUM_CLASSES * POTSDAM_CLASS_PROPORTIONS[i])
    for i in range(NUM_CLASSES)
]
POTSDAM_CLASS_WEIGHTS = [x / sum(POTSDAM_CLASS_WEIGHTS) for x in POTSDAM_CLASS_WEIGHTS]

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
    ),
    backbone=dict(
        type="MambaVisionBackbone",
        variant="mambavision_s",
        pretrained=None,
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=POTSDAM_CLASS_WEIGHTS,
            avg_non_ignore=True,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=POTSDAM_CLASS_WEIGHTS,
            avg_non_ignore=True,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
