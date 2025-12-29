
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
python process.py --src data/test/images --model_main model/best.pt --model_equipment model/equ.pt --out out
```

**参数说明：**
- `--src`: 输入图片目录路径
- `--model_main`: 主模型路径（用于检测人员和安全网）
- `--model_equipment`: 设备模型路径（用于检测救生衣）
- `--out`: 输出目录路径（默认为 `out_fused`）


### 3. 训练模型（如需重新训练）
```bash
python train.py
```

---

##  输出文件格式

运行 `process.py` 后，会在输出目录中为每张输入图片生成两个文件：

### 1. 可视化图片
- **文件名**: 与原图同名（如 `image.png`）
- **内容**: 在原图上绘制检测结果的可视化图像
  - 红色矩形框：护轮坎区域
  - 绿色矩形框：安全的人员（未在护轮坎内或已穿戴救生衣/有安全网）
  - 红色矩形框：报警的人员（在护轮坎内且未穿戴救生衣/无安全网）
  - 标签：显示人员ID和救生衣状态（如 `P0 vest:True`）

### 2. JSON 结果文件
- **文件名**: 原图名去掉扩展名 + `.json`（如 `image.json`）
- **格式**: JSON 对象，包含以下字段：

```json
{
  "image": "image.png",                    // 图片文件名
  "curb": [x1, y1, x2, y2],                // 护轮坎区域坐标 [左上x, 左上y, 右下x, 右下y]
  "persons": [                             // 检测到的人员数组
    {
      "id": 0,                             // 人员ID
      "in_curb": true,                     // 是否在护轮坎区域内
      "lifevest": {
        "present": false,                  // 是否检测到救生衣
        "source": "yolo_equipment"         // 检测来源（"yolo_equipment" 或 "none"）
      },
      "safe_net_present": false,           // 是否检测到安全网
      "alarm": true,                       // 该人员是否触发报警
      "reasons": [                         // 报警原因数组
        "in_curb_without_net",             // 在护轮坎内但无安全网
        "in_curb_without_lifevest"         // 在护轮坎内但未穿救生衣
      ]
    }
  ],
  "alarm": true,                           // 整张图片是否触发报警（任一人员报警则为true）
  "alarm_reasons": [                       // 所有报警原因（汇总）
    "in_curb_without_net",
    "in_curb_without_lifevest"
  ]
}
```

**报警规则：**
- 当人员在护轮坎区域内（`in_curb: true`）时：
  - 如果没有安全网（`safe_net_present: false`）→ 触发报警：`in_curb_without_net`
  - 如果没有救生衣（`lifevest.present: false`）→ 触发报警：`in_curb_without_lifevest`
- 如果人员不在护轮坎内或已满足安全条件，`reasons` 为 `["safe"]`，`alarm` 为 `false`

**统计输出文件：**
可以使用 `stat_alarm.py` 脚本统计输出目录中所有 JSON 文件的报警情况：

```bash
python stat_alarm.py out_warn
```

---

##  模型说明

- **选用模型**：YOLOv8n（轻量级，适合实时检测）
- **训练数据**：基于提供的码头作业图片进行标注（YOLO格式）
- **优化方向**：
  - 针对小目标检测进行特征增强
  - 使用数据增强应对复杂光照
  - 结合区域掩码，仅检测护轮坎划定区域

