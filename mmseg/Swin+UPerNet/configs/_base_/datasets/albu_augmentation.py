
dataset_type = 'COCOCustomDataset'
data_root = '/opt/ml/input/data/mmseg/'

img_norm_cfg = dict(
    # train dataset에 대해 mean/std 계산한 내용 적용 
    mean=[106.75154, 112.074585, 117.308205], std=[55.021236, 52.829964, 53.688843], to_rgb=True)

crop_size = (512, 512)

albu_transform = [
    dict(type='VerticalFlip', p=0.1),
    dict(type='HorizontalFlip', p=0.3),
    dict(type='OneOf', transforms=[
        dict(type='GaussNoise', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='Blur', p=1.0)
    ], p=0.1),
    dict(type='OneOf', 
         transforms=[
             dict(type='ShiftScaleRotate', p=1.0),
             dict(type='RandomRotate90', p=1.0),
         ], p=0.1),
    dict(type='ColorJitter', 
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.1)
]

image_size = 512
size_min, size_max = map(int, (image_size * 0.5, image_size * 1.5))

multi_scale = [(x, x) for x in range(size_min, size_max + 1, 32)]
multi_scale_test = [(x, x) for x in range(size_min, size_max + 1, 256)]
multi_scale_val = [(x, x) for x in range(size_min, size_max + 1, 512)]
multi_scale_light = [(512, 512), (768, 768), (1024, 1024)]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize',
         img_scale=multi_scale,
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Albu',
         transforms=albu_transform),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=multi_scale_light,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=multi_scale_light, multiscale_mode='value', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4, 
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True, 
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir='images/test',
        pipeline=test_pipeline))

