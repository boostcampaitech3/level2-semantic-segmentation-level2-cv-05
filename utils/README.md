## Utils

#### 데이터 시각화
- 파일: data_visualization.ipynb
- 설명
  - 경로는 ~/input/code에 두고 사용하세요.
  - 구분을 더 명확하게 하기 위해 class_dict.csv 변경하여 함께 업로드 합니다.

### remasking.py
- mask data의 보정을 위한 파일
- 512 by 512 사이즈의 마스크의 노이즈를 제거해 주는 기능

### softvoting.py
- softvoting을 위한 파일
- .pkl 확장자의 pickles 파일을 읽어서 softvoting을 진행한다
- 624, 11, 512, 512 shape를 가지고 있어야 제대로 작동한다
