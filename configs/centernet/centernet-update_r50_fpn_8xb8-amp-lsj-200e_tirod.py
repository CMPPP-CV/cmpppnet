_base_ = ['../_base_/datasets/tirod.py',
          '../_base_/default_runtime.py',
          '../_base_/schedules/schedule_2x.py']

# image_size = (1024, 1024)
# batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        # batch_augments=batch_augments
        ),
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='CTResNetNeck',
        in_channels=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=False),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=13,
        in_channels=64,
        feat_channels=256,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        # loss_wh=dict(type='L2Loss', loss_weight=0.1),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)
    ),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

train_cfg = dict(max_epochs=28, val_interval=2)

dataset_type = 'TirodDataset'
data_root = 'data/TiROD/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        # The cropped images are padded into squares during training,
        # but may be less than crop_size.
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    # Make sure the output is always crop_size.
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset = dict(
        _delete_=True,
        type='RepeatDataset',
        times=16,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='combined_annotations/train.json',
            data_prefix=dict(img='combined_images/train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args
        )
    )
    )
# Enable automatic-mixed-precision training with AmpOptimWrapper.
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(
#         type='SGD', lr=0.01 * 4, momentum=0.9, weight_decay=0.00004),
#     paramwise_cfg=dict(norm_decay_mult=0.))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=28,
        by_epoch=True,
        milestones=[18, 24],  # the real step is [18*5, 24*5]
        gamma=0.1)
]

load_from = '/net/milz/riedlinger/poisson_point_process/checkpoints/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth'

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=64)
