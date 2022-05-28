# pretrained swin transformer
```
# mmsegmentation 폴더 내부 경로에서
python tools/model_converters/swin2mmseg.py [기존 checkpoint 경로].pth [dest이름].pth
```
- 이후, config 파일에서 checkpoint 파일로 [dest이름].pth를 설정하고 학습 진행
