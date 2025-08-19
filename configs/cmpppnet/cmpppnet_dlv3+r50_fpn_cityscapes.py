_base_ = [
    '../_base_/models/cmppp_dlv3+r50_fpn.py',
    '../_base_/datasets/cityscapes_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=70, val_interval=70)
# default_scope = 'mmdet'

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=300),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90,
        by_epoch=True,
        milestones=[30, 50],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
    )