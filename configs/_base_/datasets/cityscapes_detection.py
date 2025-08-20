# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/cfs/projects/milz/datasets_robin/cityscapes/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/segmentation/cityscapes/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#          'data/': 's3://openmmlab/datasets/segmentation/'
#      }))
backend_args = None

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
    dict(
        type='Resize',
        # scale=[(2048, 800), (2048, 1024)],
        scale=(1024, 512),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='/work/riedlinger/projects/kira/point_processes_seg/mmdetection/data/cityscapes/instancesonly_filtered_gtFine_train.json',
            data_prefix=dict(img='leftImg8bit/train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/work/riedlinger/projects/kira/point_processes_seg/mmdetection/data/cityscapes/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img='leftImg8bit/val/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/work/riedlinger/projects/kira/point_processes_seg/mmdetection/data/cityscapes/instancesonly_filtered_gtFine_val.json',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

# vis_backends = [dict(type='LocalVisBackend'),
#                 #  dict(type='WandbVisBackend',
#                 #       init_kwargs={'project': 'PPP Object Detection'})
#                  ]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')