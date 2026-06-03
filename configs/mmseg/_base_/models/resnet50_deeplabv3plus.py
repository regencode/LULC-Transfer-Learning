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
        type="ResNetV1c",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        norm_eval=False,
        contract_dilation=True,
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnet50_v1c"),
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=48,
        num_classes=6,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=POTSDAM_CLASS_WEIGHTS,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=POTSDAM_CLASS_WEIGHTS,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
