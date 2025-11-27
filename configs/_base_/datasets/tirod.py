dataset_type = 'TirodDataset'
data_root = 'data/TiROD/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=
        dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='combined_annotations/train.json',
                data_prefix=dict(img='combined_images/train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args
        )
        # type="ConcatDataset",
        # datasets=[
        #     dict(
        #         type=dataset_type,
        #         data_root=data_root+f'Domain{i}/High/',
        #         ann_file='annotations/train.json',
        #         data_prefix=dict(img='images/train'),
        #         filter_cfg=dict(filter_empty_gt=True, min_size=32),
        #         pipeline=train_pipeline,
        #         backend_args=backend_args
        #     )
        #     for i in range(1,6)
        # ]+
        # [
        #     dict(
        #         type=dataset_type,
        #         data_root=data_root+f'Domain{i}/Low/',
        #         ann_file='annotations/train.json',
        #         data_prefix=dict(img='images/train'),
        #         filter_cfg=dict(filter_empty_gt=True, min_size=32),
        #         pipeline=train_pipeline,
        #         backend_args=backend_args
        #     )
        #     for i in range(1,6)
        # ]
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=
        dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='combined_annotations/val.json',
                data_prefix=dict(img='combined_images/val/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args
        )
    #     dict(
    #     type="ConcatDataset",
    #     datasets=[
    #         dict(
    #             type=dataset_type,
    #             data_root=data_root+f'Domain{i}/High/',
    #             ann_file='annotations/val.json',
    #             data_prefix=dict(img='images/val'),
    #             filter_cfg=dict(filter_empty_gt=True, min_size=32),
    #             pipeline=train_pipeline,
    #             backend_args=backend_args
    #         )
    #         for i in range(1,6)
    #     ]+
    #     [
    #         dict(
    #             type=dataset_type,
    #             data_root=data_root+f'Domain{i}/Low/',
    #             ann_file='annotations/val.json',
    #             data_prefix=dict(img='images/val'),
    #             filter_cfg=dict(filter_empty_gt=True, min_size=32),
    #             pipeline=train_pipeline,
    #             backend_args=backend_args
    #         )
    #         for i in range(1,6)
    #     ],
    # )
)
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type="ConcatDataset",
#         datasets=[
#             dict(
#                 type=dataset_type,
#                 data_root=data_root+f'Domain{i}/High/',
#                 ann_file='annotations/test.json',
#                 data_prefix=dict(img='images/test'),
#                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
#                 pipeline=test_pipeline,
#                 backend_args=backend_args
#             )
#             for i in range(1,6)
#         ]+
#         [
#             dict(
#                 type=dataset_type,
#                 data_root=data_root+f'Domain{i}/Low/',
#                 ann_file='annotations/test.json',
#                 data_prefix=dict(img='images/test'),
#                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
#                 pipeline=test_pipeline,
#                 backend_args=backend_args
#             )
#             for i in range(1,6)
#         ],
#     ),
# )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
                type=dataset_type,
                data_root=data_root+f'Domain1/High/',
                ann_file='annotations/test.json',
                data_prefix=dict(img='images/test'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=test_pipeline,
                backend_args=backend_args
            )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/work/riedlinger/projects/kira/cmpppnet/data/TiROD/combined_annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# vis_backends = [dict(type='LocalVisBackend'),
#                 #  dict(type='WandbVisBackend',
#                 #       init_kwargs={'project': 'PPP Object Detection'})
#                  ]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')