_base_ = [
    '../_base_/models/cmppp_dlv3+r50_fpn.py',
    '../_base_/datasets/cityscapes_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]



train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=128, val_interval=5)
classes = ('person', 'car')
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        dataset=dict(
            metainfo=dict(classes=classes))
    )
)
val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )
test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )
# default_scope = 'mmdet'

model=dict(
    backbone=dict(
        decode_head=dict(num_classes=3 + len(classes)),
        auxiliary_head=dict(num_classes=3 + len(classes))
    ),
    bbox_head=dict(
        type='CMPPPHead',
        pooling_size=16,
        in_channels=3 + len(classes),
        num_classes=len(classes),
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
        milestones=[96, 116],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-2, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
    )

# load_from='/net/milz/riedlinger/poisson_point_process/checkpoints/deeplabv3plus_r50_backbone.pth'
load_from='/work/riedlinger/projects/kira/cmpppnet/work_dirs/cmpppnet_dlv3+r50_fpn_cityscapes_3classes/epoch_10.pth'