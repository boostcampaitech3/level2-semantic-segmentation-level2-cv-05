# Mask2Former
본 폴더는 Facebook research Mask2Former를 기반으로 만들어졌습니다.

:closed_book: https://github.com/facebookresearch/Mask2Former

### 0. 환경 정보
```
torch==1.7.1
cudatoolkit==11.0
detectron==0.5+cu110
```


### 1. 환경 세팅


1. conda create --name [NAME] --clone base
2. cudatoolkitdev=11.0 설치
   * Aistage에서의 Cuda Setting에 대해서는 다음의 글을 참고
   * [Conda Cuda setting](https://kyubumshin.github.io/2022/04/23/tip/conda-cuda-%EC%84%A4%EC%B9%98/)
3. detectron2 설치
```
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```
4. Mask2former git clone
```
git clone https://github.com/facebookresearch/Mask2Former
```
5. pixel decoder 설정
```
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
6. detectron2 package 코드 수정
```
conda/envs/[NAME]/lib/python3.8/site-packages/detectron2/project/point_rend/point_features.py
```
* 48 line 수정
```
output = F.grid_sample(input.float(), 2.0 * point_coords - 1.0, **kwargs)
```
