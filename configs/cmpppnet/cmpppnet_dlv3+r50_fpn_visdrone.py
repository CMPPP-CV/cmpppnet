_base_ = [
    '../_base_/models/cmppp_dlv3+r50_fpn.py',
    '../_base_/datasets/visdrone.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

backend_args = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=128, val_interval=5)

dataset_type = 'VisdroneDataset'
data_root = '/net/milz/riedlinger/poisson_point_process/visdrone_data/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(1024,512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'train/coco_annotations.json',
            data_prefix=dict(img='train/images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))
# default_scope = 'mmdet'

model=dict(
    backbone=dict(
        decode_head=dict(num_classes=3 + 11),
        auxiliary_head=dict(num_classes=3 + 11)
    ),
    bbox_head=dict(
        type='CMPPPHead',
        pooling_size=16,
        in_channels=3 + 11,
        num_classes=11,
    )
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=300),
    dict(
        type='MultiStepLR',
        begin=0,
        end=128,
        by_epoch=True,
        milestones=[114, 124],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-2, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
    )

load_from='/net/milz/riedlinger/poisson_point_process/checkpoints/deeplabv3plus_r50_backbone.pth'
# load_from='/work/riedlinger/projects/kira/cmpppnet/work_dirs/cmpppnet_dlv3+r50_fpn_visdrone/epoch_72.pth'