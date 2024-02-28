![middle](https://capsule-render.vercel.app/api?type=cylinder&color=0147FF&height=150&section=header&text=Wassup&fontColor=FFFFFF&fontSize=70&animation=fadeIn&fontAlignY=55)

### Team
- [DoYeon Kim](https://github.com/electronicguy97) - Team Leader
- [HyunJun Kang](https://github.com/)
- [Jongseong Kim](https://github.com/JamieSKinard)
- [Chaewook Lee](https://github.com/leecw12)
- [HaNeul Pyeon](https://github.com/Haneul1002)

### PROJECT
저희는 50만건의 동양인 감정을 사용했습니다.<br>
행복, 화남, 불안, 당황, 평시, 아픔, 슬픔 총 7가지의 감정을 활용했습니다.<br>
전처리로는 YOLOv8-face를 사용하여 얼굴만 잘라 사용하여 정확도와 학습시간을 얻었습니다.<br>
2-stage 모델로는 Repvgg, VIT(Vision Transformer)와 YOLO를 사용했고, 1-stage 모델로는 YOLO 사용했습니다.<br>
Streamlit으로도 여러기능을 만들었으니 구경해주세요.<br>

![image](https://github.com/electronicguy97/est_wassup_03/assets/103613730/41417652-dea9-4123-a3d9-5332af6f4bc6)

### 장비
GPU server : A-100 4대(AWS)
OS : Linux
Language : Python

### 실험보고서
|실험보고서|발표자료|
|---|---|
|||

### 사용 패키지 설치방법
```bash
git clone https://github.com/JamieSKinard/est_wassup_03.git
cd est_wassup_03
pip install -r requirements.txt
pip install -e .
```
or
```bash
# using conda
conda env create -f env.yaml
```

### 전처리방법
```bash
# check default path
python preprocess/preprocess.py --data-dir {your_data_path}
```
YOLOv8-face모델을 사용해서 BBOX기준 얼굴 부분만 잘라서 새로운 위치에 데이터를 생성합니다. 이것을 통해서 사진의 크기를 줄여 학습시간을 단축 시킬 수 있습니다.

### 2Stage Model(RepVgg, VIT, YOLO) 사용방법
```bash
python main.py --data-dir {your_data_path}
### Check defult
### Change model -> choice(defalut = RepVGG, VIT)
python main.py -mn VIT --data-dir {your_data_path} -mp {your_model} -mn {Reppvgg or VIT}
```
결과와 모델은 Models(defalut)/{선택모델} 안에 들어 있습니다.

### YOLO(1 Stage) 사용방법
디렉토리에서 box_labeling_yolov8.ipynb를 통해 BBOX Label이 가능합니다. 하셨다면 YOLO.ipynb에서 진행하시면 됩니다.

### 평가방법
```bash
python eval.py --data-dir {your_test_folder_path} -mp {your_model_path} -mn {Repvgg, VIT}
```
test 할 때 사용할 데이터 경로 설정 후 학습 시켰던 모델과 경로를 설정하시면 됩니다.
평가지표로는 f1, R2, Precision, recall 사용했습니다.

### 결과
||YOLO(1Stage)|YOLO(2Stage)|ReppVgg|VIT|
|---|---|---|---|---|
|val_loss|0.233|0.533|1.470|1.251|
|train_acc|||78.3%|72.2%|
|val_acc||73.9%|68.7%|62.5%|


<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src = "https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">