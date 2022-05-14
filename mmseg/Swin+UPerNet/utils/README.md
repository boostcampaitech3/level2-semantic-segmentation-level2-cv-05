## 📝 mask_viz.ipynb
- 생성된 mask를 원본 이미지와 함께 시각화한다. 

## 📝 calculate_mean_std.py
- train dataset의 mean/std 값을 계산한다. 

## 📂 copy_paste

- 참고 : [Copy-Paste-for-Semantic-Segmentation](https://github.com/qq995431104/Copy-Paste-for-Semantic-Segmentation)
```
📂 copy_paste
├── 📝 copy_paste.py 
└── 📝 get_coco_mask.py
```
아래의 순서대로 파일을 실행시킨다. 
#### 1) 📝 get_coco_mask.py
- train dataset의 mask 이미지를 생성한다.  

#### 2) 📝 copy_paste.py 
- copy paste augmentation이 적용된 image와 mask를 생성한다.
