_base_ = [
    '../_base_/models/cmppp_dlv3+r50_fpn.py',
    '../_base_/datasets/cityscapes_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]



train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=192, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    num_workers=4
)
# default_scope = 'mmdet'

model=dict(
    bbox_head=dict(
        type='CMPPPHead',
        pooling_size=16,
    )
)

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=300),
    dict(
        type='MultiStepLR',
        begin=0,
        end=128,
        by_epoch=True,
        milestones=[128],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-2, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
    )

load_from='/net/milz/riedlinger/poisson_point_process/checkpoints/deeplabv3plus_r50_backbone.pth'