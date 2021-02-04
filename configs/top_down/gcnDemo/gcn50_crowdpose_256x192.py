log_level = 'INFO'
# load_from should be the trained cfa weight
load_from ='work_dirs/cfa50_crowdpose_256x192/model_best.pth'
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(type='MyCheckpointHook', interval=10, with_indicator=False)
evaluation = dict(interval=10, metric='mAP', key_indicator='AP', start_epoch=100)

optimizer = dict(
    type='Adam',
    lr=1e-3,
    # paramwise_cfg is used to freeze backbone and keypoint_head params
    paramwise_cfg=dict(
        custom_keys={'backbone.': dict(lr_mult=0.0),
                     'keypoint_head.': dict(lr_mult=0.0)}
    )
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

# model settings
model = dict(
    type='TopDownGCN',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopDownCFAHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'], 
        extra=dict(cfa_out=True)
    ),
    gcn_head=dict(
        adj=[[12, 13],[13,0],[13,1],[0,2],[2,4],[1,3],[3,5],[13, 7],[13,6],[7,9],[9,11],[6,8],[8,10]],
        num_joints=channel_cfg['num_output_channels'],
        hid_dim=[128, 128, 128, 128, 128]
    ),
    train_cfg=dict(loss_weights=[0.3, 0.5, 1, 1]),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11),
    loss_pose=dict(type='JointsMSELoss', use_target_weight=True, crit_type='L1Loss'), 
    extra=dict(
        backbone_acc=True,
        # only_inference means not update backbone params and not calculate backbone heatmap mse_loss
        only_inference=True, 
        # test backbone ap
        backbone_test=False))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    crowd_matching=False,
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    bbox_thr=1.0,
    use_gt_bbox=False,
    image_thr=0.0,
    bbox_file='data/crowdpose/annotations/'
    'det_for_crowd_test_0.1_0.5.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=6,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/crowdpose'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_trainval.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline))
