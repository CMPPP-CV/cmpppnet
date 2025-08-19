norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True)
num_classes = 8

bb_model = dict(
    type='SegEncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        frozen_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        act_cfg=dict(negative_slope=0.1, type='LeakyReLU'),
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3 + num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        act_cfg=dict(negative_slope=0.1, type='LeakyReLU'),
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3+num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model = dict(
    type='CMPPPNet',
    data_preprocessor=data_preprocessor,
    backbone=bb_model,
    neck=dict(
        type='CMPPPNeck'),
    bbox_head=dict(
        type='CMPPPHead',
        in_channels=3 + num_classes,
        num_classes=num_classes,
        loss_center_heatmap=dict(type='PPPLoss', loss_weight=1.0),
        loss_classification=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=1.0),
    )
)