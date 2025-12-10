# 图像自动标注系统 (Auto-Annotation System)

基于YOLO的图像自动标注系统，通过少量标注数据训练模型，自动标注大规模数据集。

## 📋 功能特性

- ✅ 支持多种YOLO版本 (YOLOv5/v8/v11)
- ✅ 自动数据集划分和组织
- ✅ 模型训练和验证
- ✅ 批量图像自动标注
- ✅ 置信度分级（高/中/低）
- ✅ 完整的命令行工具
- ✅ 灵活的配置系统

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将您的初始标注数据放在 `data/raw/` 目录下：

```
data/raw/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── labels/
    ├── img1.txt
    ├── img2.txt
    └── ...
```

标注格式为YOLO格式（每行：`class_id x_center y_center width height`，坐标归一化）

### 3. 运行完整流程

```bash
python run_pipeline.py --mode full --raw-data data/raw --unlabeled data/unlabeled/images
```

## 📖 详细使用说明

### 方式一：分步执行

#### 步骤1：准备数据集

```bash
python scripts/prepare_data.py --data-dir data/raw --output-dir data --split-ratio 0.2
```

参数说明：
- `--data-dir`: 原始数据目录
- `--output-dir`: 输出目录
- `--split-ratio`: 验证集比例（默认0.2）
- `--seed`: 随机种子（默认42）

#### 步骤2：训练模型

```bash
python scripts/train_model.py --config config/config.yaml --data config/dataset_config.yaml
```

参数说明：
- `--config`: 主配置文件
- `--data`: 数据集配置文件
- `--epochs`: 训练轮数（可选，覆盖配置）
- `--batch-size`: 批次大小（可选）
- `--device`: 设备（cuda/cpu，可选）

#### 步骤3：自动标注

```bash
python scripts/auto_label.py --model models/trained/train/weights/best.pt --images data/unlabeled/images --output output/predictions
```

参数说明：
- `--model`: 训练好的模型路径
- `--images`: 待标注图像目录
- `--output`: 输出目录
- `--conf-threshold`: 置信度阈值（可选）

### 方式二：使用Pipeline

```bash
# 完整流程
python run_pipeline.py --mode full

# 仅准备数据
python run_pipeline.py --mode prepare --raw-data data/raw

# 仅训练
python run_pipeline.py --mode train

# 仅标注
python run_pipeline.py --mode annotate --model path/to/model.pt --unlabeled data/unlabeled/images
```

## ⚙️ 配置说明

### 主配置文件 (`config/config.yaml`)

```yaml
training:
  model_type: "yolov8"        # yolov5, yolov8, yolov11
  model_size: "m"             # n, s, m, l, x
  epochs: 100
  batch_size: 16
  img_size: 640
  device: "cuda"              # cuda, cpu, mps

auto_annotation:
  confidence_threshold: 0.6   # 预测置信度阈值
  review_threshold: 0.5       # 低于此值需人工复审
```

## 📊 输出结果

### 训练输出

```
models/trained/train/
├── weights/
│   ├── best.pt          # 最佳模型
│   └── last.pt          # 最后一个epoch
└── results.png          # 训练曲线
```

### 自动标注输出

```
output/predictions/
├── labels/
│   ├── high_conf/       # 高置信度 (>0.7)
│   ├── medium_conf/     # 中等置信度 (0.5-0.7)
│   └── low_conf/        # 低置信度 (<0.5, 需复审)
└── statistics.json      # 统计信息
```

## 📈 置信度分级说明

- **高置信度 (>0.7)**: 可直接使用
- **中等置信度 (0.5-0.7)**: 建议抽查验证
- **低置信度 (<0.5)**: 需要人工复审

## 🔧 高级功能

### 自定义配置

编辑 `config/config.yaml` 调整训练参数：

```yaml
training:
  epochs: 200              # 增加训练轮数
  batch_size: 32           # 增大批次
  patience: 50             # Early stopping耐心值
```

### 使用不同YOLO版本

```yaml
training:
  model_type: "yolov11"    # 切换到YOLOv11
  model_size: "l"          # 使用大模型
```

## 📁 项目结构

```
model_train/
├── config/              # 配置文件
├── data/                # 数据集
│   ├── raw/            # 原始数据
│   ├── train/          # 训练集
│   ├── val/            # 验证集
│   └── unlabeled/      # 待标注数据
├── models/              # 模型
│   ├── pretrained/     # 预训练权重
│   └── trained/        # 训练后模型
├── output/              # 输出结果
├── src/                 # 源代码
│   ├── data_processor.py
│   ├── trainer.py
│   ├── predictor.py
│   └── auto_annotator.py
├── scripts/             # 命令行脚本
└── run_pipeline.py      # 完整流程脚本
```

## 🐛 常见问题

### Q: CUDA out of memory

A: 减小batch_size或使用更小的模型：

```yaml
training:
  batch_size: 8
  model_size: "s"
```

### Q: 训练速度慢

A: 检查是否使用GPU：

```python
import torch
print(torch.cuda.is_available())  # 应该返回True
```

### Q: 自动标注质量不高

A: 尝试：
1. 增加训练数据量
2. 增加训练轮数
3. 使用更大的模型
4. 调整置信度阈值

## 📝 使用示例

### 示例1：训练车辆检测模型

```bash
# 1. 准备100张标注好的车辆图像
python scripts/prepare_data.py --data-dir data/raw

# 2. 训练模型
python scripts/train_model.py --epochs 150

# 3. 标注5000张新图像
python scripts/auto_label.py --model models/trained/train/weights/best.pt --images data/unlabeled/images
```

### 示例2：使用预训练模型微调

配置文件中设置：

```yaml
training:
  pretrained: true  # 使用COCO预训练权重
  epochs: 50        # 较少的轮数即可
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请提交Issue或联系项目维护者。

---

**版本**: 1.0.0  
**最后更新**: 2025-12-08