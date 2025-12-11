
#  护轮坎人员闯入报警模型项目

##  项目概述
本项目基于宝钢梅山钢铁原料码头现场采集的视频数据，建立了一个**人员闯入报警模型**，用于检测码头护轮坎区域在未穿戴救生衣、安全网未设置等危险情况下的人员闯入行为，实现实时预警与安全监控。


##  子课题选择：子课题2
- **任务目标**：使用提供的图片数据针对划定区域建立人员闯入报警模型。
- **模型类型**：基于深度学习的目标检测模型（如 YOLO、Faster R-CNN 等）。
- **开发语言**：Python

---

##  项目结构

```
YOLO/
├── data/
│   ├── train/images
│   ├── train/labels
│   ├── valid/images
│   ├── valid/labels
│   └── test/images
├── data.yaml
├── yolov8n.pt  (可选)
└── train.py   ← 放这里

```

---

##  快速开始

### 1. 环境配置
建议使用 Python 3.8+，安装依赖：

```bash
conda create -n yoloenv python=3.10 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ultralytics

```

### 2. 运行推理
使用 `process.py` 进行推理：

```bash
python process.py --src data/test/images --model runs/detect/train/weights/best.pt --out out --conf 0.25
```


### 3. 训练模型（如需重新训练）
```bash
python train.py
```

---

##  模型说明

- **选用模型**：YOLOv8n（轻量级，适合实时检测）
- **训练数据**：基于提供的码头作业图片进行标注（YOLO格式）
- **优化方向**：
  - 针对小目标检测进行特征增强
  - 使用数据增强应对复杂光照
  - 结合区域掩码，仅检测护轮坎划定区域

