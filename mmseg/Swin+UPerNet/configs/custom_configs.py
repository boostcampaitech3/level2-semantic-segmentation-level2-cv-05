_base_ = [
    './_base_/models/[model 이름].py', './_base_/datasets/[dataset 이름].py',
    './_base_/custom_runtime.py', './_base_/schedules/custom_schedule.py'
]

# custom_schedule.py에서 원하는 optimizer/lr scheduler 을 제외한 나머지는 주석 처리 필요 