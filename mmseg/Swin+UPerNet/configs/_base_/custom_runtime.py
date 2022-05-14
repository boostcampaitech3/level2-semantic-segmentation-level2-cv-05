date = '2022-00-00'

log_config = dict(interval=50, 
                  hooks=[
                    dict(type='TextLoggerHook', by_epoch=False),      
                    dict(type='WandbLoggerHook',interval=50,
                    init_kwargs=dict(
                    project='[프로젝트명]',
                    entity = '[wandb id]',
                    name = f"{date}-[실험 내용]" 
                    ),
                    ),])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
cudnn_benchmark = True

work_dir = f'./work_dirs/[실험 내용]/{date}'
seed = 2022
gpu_ids = 0