![middle](https://capsule-render.vercel.app/api?type=cylinder&color=0147FF&height=150&section=header&text=Wassup&fontColor=FFFFFF&fontSize=70&animation=fadeIn&fontAlignY=55)
### PROJECT

### Team
- [DoYeon Kim](https://github.com/electronicguy97) - Team Leader
- [HyunJun Kang](https://github.com/)
- [Jongseong Kim](https://github.com/JamieSKinard)
- [Chaewook Lee](https://github.com/leecw12)
- [HaNeul Pyeon](https://github.com/Haneul1002)

### 실험보고서
|실험보고서|발표자료|
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
python preprocess/preprocess.py --data-dir {your_data_path}
```

### How to Learn(2Stage Model)
```bash
python main.py --data-dir {your_data_path}
### Check defult
### Change model -> choice(defalut = RepVGG, VIT)
python main.py -mn VIT --data-dir {your_data_path}
```

### How to Learn(1Stage Model)
pass



<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src = "https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
