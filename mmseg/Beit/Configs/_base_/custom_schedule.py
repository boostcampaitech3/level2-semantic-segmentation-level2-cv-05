# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.01)
# f16 
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=None)
fp16 = dict()

# runtime
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=327,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

total_epochs = 40
