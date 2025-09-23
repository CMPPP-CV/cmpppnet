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
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3+num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
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
        pooling_size=16,
    )
)