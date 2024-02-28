![middle](https://capsule-render.vercel.app/api?type=cylinder&color=0147FF&height=150&section=header&text=Wassup&fontColor=FFFFFF&fontSize=70&animation=fadeIn&fontAlignY=55)

### Team
- [DoYeon Kim](https://github.com/electronicguy97) - Team Leader
- [HyunJun Kang](https://github.com/)
- [Jongseong Kim](https://github.com/JamieSKinard)
- [Chaewook Lee](https://github.com/leecw12)
- [HaNeul Pyeon](https://github.com/Haneul1002)

### PROJECT
We started with 500,000 pieces of data.<br>
It has 7 classes: happy, anger, anxiety, embarrass, normal, pain, and sad.<br>
As preprocessing, we reduced the number of data or used YOLOv8 to crop only the faces. Cropped photos to reduce storage capacity to use less GPU<br>
Repvgg and VIT (Vision Transformer, YOLO) were used as 2-stage models, and YOLO was used as 1-stage model.<br>
We have implemented many functions using Streamlit, so please check it out.<br>

![image](https://github.com/electronicguy97/est_wassup_03/assets/103613730/41417652-dea9-4123-a3d9-5332af6f4bc6)

### experiment report
|experiment report|presentation|
|---|---|
|||
|||

### How to Install
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

### How to Pretreatment
```bash
# check default path
python preprocess/preprocess.py --data-dir {your_data_path}
```
Create a cropped photo after face recognition with the yolo8n-face model

### How to Learn(2Stage Model)
```bash
python main.py --data-dir {your_data_path}
### Check defult
### Change model -> choice(defalut = RepVGG, VIT)
python main.py -mn VIT --data-dir {your_data_path} -mp {your_model} -mn {Reppvgg or VIT}
```
Saved in Models folder/{your_choice_model}

### How to Learn(1Stage Model)
Preprocessing is possible with box_labeling_yolov8.ipynb in the folder called preprocess.
and Go to the file named YOLO.ipynb and Just Shift + F5

### How to evaluation
```bash
python eval.py --data-dir {your_test_folder_path}
```
We used f1, R2, Precision, and recall as metrics.

### result
||YOLO(1Stage)|YOLO(2Stage)|ReppVgg|VIT|
|---|---|---|---|---|
|val_loss|0.233|0.533|1.470|1.251|
|train_acc|||78.3%|72.2%|
|val_acc||73.9%|68.7%|62.5%|


<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src = "https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
