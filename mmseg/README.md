# mmsegmentation 환경설정

```
conda create -n [가상환경이름] python=3.8 -y
conda init --all
source ./.zshrc
conda activate [가상환경이름]
```
```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
conda install cudatoolkit-dev=11.0 -c conda-forge
apt-get install g++

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html

pip install mmsegmentation 
git clone https://github.com/open-mmlab/mmsegmentation.git 

cd mmsegmentation
pip install -r requirements.txt
python setup.py install 
```
```
# optional
pip install wandb
```
- 참고 : [Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)    



# train

```
# cd mmsegmentation 
python tools/train.py [config 파일 경로]
```

# inference
- inference.ipynb 실행
- TTA 진행 시, 아래와 같이 코드 수정 필요
```
multi_scale = [(512, 512), (1024, 1024), ... ]


cfg = Config.fromfile(CONFIG_PATH)
root=TEST_IMAGES_PATH

# dataset config 수정
cfg.data.test.img_dir = root
cfg.data.test.pipeline[1]['img_scale'] = multi_scale # 수정 필요 
cfg.data.test.test_mode = True
cfg.data.samples_per_gpu = 1
cfg.work_dir = WORK_DIR
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{ITER}.pth')
```

