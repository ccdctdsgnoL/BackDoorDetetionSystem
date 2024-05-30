# 神经网络的后门检测系统

## 1. 项目介绍

神经网络的后门检测系统

## 2. 软件架构

软件架构说明
```
BackDoorDetetionSystem
├─data  用来存放数据集
├─experiments  训练模型存放位置
├─models  模型代码
├─utils  工具类
│  ├─BackdoorStyleGenerator.py  后门样式生成器
│  ├─BackDoorStyleTools.py  后门样式工具
│  ├─BDDSystemGUI.py  GUI界面
│  ├─ColorPrint.py  彩色打印
│  ├─ImageDisplay.py  图片显示
│  ├─ModelSelect.py  模型选择
│  ├─showTensor.py  图像化显示tensor数据图片
│  ├─tensor2img.py  tensor转换为图片
│  └─TestedModel.py  被测模型类
├─BackDoorDetetionSystem.py  主界面
├─BDDSystem.py  后门检测系统主要代码
├─ModelGen.py  后门模型生成代码
└─ReverseFeature.py  特征提取代码
```

## 3. 安装教程


### 环境配置
```
# packages in environment at E:\Code\Anaconda\envs\PyTorch:
#
# Name                    Version                   Build  Channel
matplotlib                3.5.1                    pypi_0    pypi
numpy                     1.21.6                   pypi_0    pypi
pillow                    9.5.0                    pypi_0    pypi
pip                       24.0               pyhd8ed1ab_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
python                    3.7.16               h6244533_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
torch                     1.8.0+cu111              pypi_0    pypi
torchaudio                0.8.0                    pypi_0    pypi
torchvision               0.9.0+cu111              pypi_0    pypi
```
```
pip install -r requirements.txt
```

### 运行
```
python BackDoorDetetionSystem.py
```

## 4. 使用说明

1. 下载数据集
2. 运行代码
3. 选择模型
4. 选择后门样式
5. 选择后门图片
6. 运行
