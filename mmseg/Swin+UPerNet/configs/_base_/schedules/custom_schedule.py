runner = dict(type='IterBasedRunner', max_iters=80000) 

checkpoint_config = dict(max_keep_ckpts=2, interval=1000, by_epoch=False) # latest.pth를 최대 2개까지 저장
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU') 


###############################
# Optimizer 
###############################

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2)) # apply fp16
fp16 = dict() 

# AdamW 
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_tab.le': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# SGD 
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# Adam
# optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)


###############################
# LR Scheduler 
###############################

# poly
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# cyclic
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 1e-4),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )