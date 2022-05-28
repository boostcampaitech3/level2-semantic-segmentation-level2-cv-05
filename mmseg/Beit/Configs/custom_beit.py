_base_ = [
    '../_base_/custom_model_beit.py', '../_base_/custom_dataset.py',
    '../_base_/custom_runtime.py', '../_base_/custom_schedule.py'
]
crop_size = (512, 512)

model = dict(
    pretrained='beit_base_convert.pth',
    backbone=dict(
        type='BEiT',
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        qv_bias=True,
        init_values=0.1,
        drop_path_rate=0.1,
        out_indices=(3, 5, 7, 11)),
    neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], num_classes=11, channels=768),
    auxiliary_head=dict(in_channels=768, num_classes=11),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(128, 128)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=327,
    warmup_ratio=0.1,
    min_lr_ratio=1e-06)
# By default, models are trained on 8 GPUs with 2 images per GPU

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    val=dict(pipeline=test_pipeline))
