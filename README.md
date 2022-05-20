# 

# [AI Tech 3기 Level 2 P Stage] Semantic Segmentation

<img width="809" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/169448649-0e2d62e9-c41f-441f-9c41-a157ce077e6d.png">


# ConVinsight 🧑‍💻

Convenience + Insight : 이용자의 편의를 찾는 통찰력

## Member
| 김나영 | 신규범 | 이정수 | 이현홍 | 전수민 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/dudskrla) | [Github](https://github.com/KyubumShin) | [Github](https://github.com/sw930718) | [Github](https://github.com/Heruing) | [Github](https://github.com/Su-minn) |
## Wrap Up Report 📑

💻 [Convinsight level2-semantic-segmentation pdf](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-05/files/8735886/wrap.up.report.pdf)   

## Final Score 🏆

- Public f1 score 0.8318 → Private f1 score 0.8024
- Public 1위 → Private 1위

![result](https://user-images.githubusercontent.com/90603530/169448783-e81e83d4-c3d7-4d98-bd58-91d18cc680cb.gif)

## Competition Process 🗓️

### Time Line

> **SeMask / Mask2Former**
> 
![규범님](https://user-images.githubusercontent.com/90603530/169448806-39dfd733-fc6b-4a9d-9bd9-6ba4780d6321.jpg)


> **Swin + UPerNet**
> 
![swin+upernet](https://user-images.githubusercontent.com/90603530/169448819-f5fb222d-ed0f-41d5-bcd6-9097f23e4e13.jpg)


> **HRNet v2 + OCR**
> 
![현홍님](https://user-images.githubusercontent.com/90603530/169448837-55a3c005-2518-4465-a9fa-fe4c8dd0745f.jpg)


> **UNet++**
> 
![수민님](https://user-images.githubusercontent.com/90603530/169448845-5f883d1f-9b52-40ef-a7fb-2bf199744a7a.jpg)


> **BEiT / Mask2Former**
> 
![정수님](https://user-images.githubusercontent.com/90603530/169448857-13ca9e21-663d-494f-ba4f-b97fa222b79b.jpg)


### Project Outline

> **SeMask / Mask2Former**
> 
- [Mask2Former 실험](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-05/issues/9)

> **Swin + UPerNet**
> 
- [Swin + UPerNet 실험](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-05/issues/16)

> **HRNet v2 + OCR**
> 
- [HRNet v2 + OCR 실험](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-05/issues/12)

> **UNet++**
> 
- [UNet++ 실험](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-05/issues/18)

> **BEiT / Mask2Former**
> 
- [BEiT 실험](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-05/issues/17)

### Folder Structure 📂

```
level2-semantic-segmentation-level2-cv-05/
│
├── 📂 Baseline
│   ├── 📝 baseline_fcn_resnet50.ipynb
│   ├── 📝 class_dict.csv
│   ├── 📝 data_visualization.ipynb
│   ├── 📝 requirements.txt
│   └── 📝 utils.py
│
├── 📂 mask2former
│   ├── 📂 config/custom
│   │ 	 └── 📝 *.yaml
│   ├── 📝 inference.py
│   ├── 📝 inference_pkl.py
│   ├── 📝 inference_tta.py
│   ├── 📝 mask_masking.py
│   ├── 📝 register_trash_dataset.py
│   ├── 📝 register_trash_dataset_fix.py
│   └── 📝 train_net.py
│
├── 📂 mmseg
│   ├── 📂 Beit
│   │ 	 └── 📂 Configs
│   │ 	      ├── 📂 _base_
│   │ 	      │    └── 📝 *.py
│   │ 	      └── 📝 custom_beit.py
│   ├── 📂 Swin+UPerNet
│   │ 	 ├── 📂 configs
│   │ 	 │    ├── 📂 _base_
│   │ 	 │    │    └── 📝 *.py
│   │ 	 │    └── 📝 custom_configs.py
│   │ 	 ├── 📂 datasets
│   │ 	 │    └── 📝 coco_custom_dataset.py
│   │ 	 ├── 📂 tools
│   │ 	 │    └── 📝 train.py
│   │ 	 └── 📂 utils
│   │ 	      ├── 📂 copy_paste
│   │ 	      │    ├── 📝 copy_paste.py
│   │ 	      │    └── 📝 get_coco_mask.py
│   │ 	      ├── 📝 calculate_mean_std.py
│   │ 	      └── 📝 mask_viz.ipynb
│   └── 📝 inference.ipynb
│
├── 📂 smp
│   ├── 📝 config.yaml
│   ├── 📝 dataset.py
│   ├── 📝 inference.py
│   ├── 📝 loss.py
│   ├── 📝 model.py
│   ├── 📝 requirements.txt
│   ├── 📝 train.py
│   ├── 📝 train.sh
│   └── 📝 utils.py
│
└── 📂 utils
    ├── 📝 EDA_nayoung.ipynb
    ├── 📝 class_dict.csv
    ├── 📝 data_visualization.ipynb
    ├── 📝 remasking.py
    ├── 📝 soft_voting.py
    └── 📝 submission_viz.ipynb
```
