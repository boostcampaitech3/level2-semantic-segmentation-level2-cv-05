log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),  
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project = '프로젝트 명',
                entity = 'entitiy 명',
                name = "실험명" )
        )
        ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1),('val',1)] # wandb train, val 확인 코드  
cudnn_benchmark = True

runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=10)
evaluation = dict(metric='mIoU', save_best='mIoU')