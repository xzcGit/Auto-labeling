# 自定义路径模式使用指南

## 概述

`train_by_category.py` 支持两种运行模式：
1. **默认模式**：处理项目内 `data/raw/` 下的所有类别
2. **自定义路径模式**：处理任意位置的所有类别（批量处理）

## 自定义路径模式

### 目录结构要求

在使用自定义路径模式前，请确保你的目录结构如下：

```
/your/custom/root/
├── category1/           # 类别1
│   ├── pre_images/      # 必需：已标注的图像
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── pre_labels/      # 必需：对应的YOLO格式标签
│   │   ├── img001.txt
│   │   ├── img002.txt
│   │   └── ...
│   └── images/          # 可选：待标注的图像
│       ├── new001.jpg
│       └── ...
├── category2/           # 类别2
│   ├── pre_images/
│   ├── pre_labels/
│   └── images/
└── category3/           # 类别3
    ├── pre_images/
    └── pre_labels/
```

### 输出结构

运行后，程序会在每个类别目录下自动创建输出：

```
/your/custom/root/
├── category1/
│   ├── pre_images/      # 原始输入（保持不变）
│   ├── pre_labels/      # 原始输入（保持不变）
│   ├── images/          # 输入：待标注图像（如果存在）
│   ├── category/        # 输出：划分后的数据集
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── val/
│   │       ├── images/
│   │       └── labels/
│   ├── models/          # 输出：训练的模型
│   │   └── train/
│   │       └── weights/
│   │           ├── best.pt
│   │           └── last.pt
│   ├── labels/          # 输出：自动生成的标签
│   │   ├── high_conf/
│   │   ├── medium_conf/
│   │   └── low_conf/
│   └── dataset_config.yaml  # 输出：数据集配置
├── category2/
│   ├── pre_images/
│   ├── pre_labels/
│   ├── category/
│   ├── models/
│   └── ...
└── category3/
    └── ...
```

## 使用方法

### 基本用法

```bash
# 处理自定义路径下的所有类别
python scripts/train_by_category.py --data-root /path/to/your/data
```

### 使用自定义配置文件

```bash
python scripts/train_by_category.py \
    --data-root /path/to/your/data \
    --config /path/to/custom/config.yaml
```

## 完整示例

### 示例1：批量训练多个类别

```bash
# 1. 准备数据结构
mkdir -p /data/industrial_meters/pointer_meter/pre_images
mkdir -p /data/industrial_meters/pointer_meter/pre_labels
mkdir -p /data/industrial_meters/digital_meter/pre_images
mkdir -p /data/industrial_meters/digital_meter/pre_labels
mkdir -p /data/industrial_meters/switch_button/pre_images
mkdir -p /data/industrial_meters/switch_button/pre_labels

# 将已标注数据放入对应目录
# pointer_meter: 50张图像 + 标签
# digital_meter: 30张图像 + 标签
# switch_button: 20张图像 + 标签

# 2. 一键批量训练所有类别
python scripts/train_by_category.py --data-root /data/industrial_meters

# 3. 查看结果
# 指针表模型：/data/industrial_meters/pointer_meter/models/train/weights/best.pt
# 数字表模型：/data/industrial_meters/digital_meter/models/train/weights/best.pt
# 开关按钮模型：/data/industrial_meters/switch_button/models/train/weights/best.pt
```

### 示例2：训练并自动标注

```bash
# 1. 准备数据（包含待标注图像）
/data/meters/
├── pointer_meter/
│   ├── pre_images/      # 50张已标注
│   ├── pre_labels/      # 50个标签
│   └── images/          # 1000张待标注
└── digital_meter/
    ├── pre_images/      # 30张已标注
    ├── pre_labels/      # 30个标签
    └── images/          # 500张待标注

# 2. 运行训练和自动标注
python scripts/train_by_category.py --data-root /data/meters

# 3. 查看自动标注结果
# 指针表标签：/data/meters/pointer_meter/labels/high_conf/
# 数字表标签：/data/meters/digital_meter/labels/high_conf/
```

### 示例3：Windows 路径

```bash
# Windows 使用反斜杠或正斜杠都可以
python scripts/train_by_category.py --data-root "D:\datasets\my_objects"

# 或者
python scripts/train_by_category.py --data-root D:/datasets/my_objects
```

## 与默认模式的对比

| 特性 | 默认模式 | 自定义路径模式 |
|------|---------|---------------|
| 数据位置 | `data/raw/` | 任意路径 |
| 输入目录名 | `images/`, `labels/` | `pre_images/`, `pre_labels/` |
| 输出位置 | 项目目录下集中管理 | 每个类别目录下 |
| 批量处理 | 支持（自动扫描） | 支持（自动扫描） |
| 待标注图像位置 | `{category}_unlabeled/images/` | `{category}/images/` |
| 适用场景 | 项目内多类别开发 | 处理外部数据或独立管理 |

## 常见问题

### Q1: 为什么要用 pre_images 和 pre_labels？

A: 为了区分输入和输出：
- `pre_images/`, `pre_labels/`：你提供的已标注数据（输入，保持不变）
- `images/`：待标注图像（输入）
- `labels/`：自动生成的标签（输出）
- `category/`：划分后的训练数据集（输出）
- `models/`：训练的模型（输出）

### Q2: 可以不提供待标注图像吗？

A: 可以。如果某个类别目录下不存在 `images/` 目录，程序会跳过该类别的自动标注步骤，只进行数据划分和模型训练。

### Q3: 训练完成后，原始数据会被修改吗？

A: 不会。`pre_images/` 和 `pre_labels/` 保持不变，所有输出都在新创建的目录中。

### Q4: 可以重复运行吗？

A: 可以。重复运行会覆盖之前的输出（`category/`, `models/`, `labels/`），但不会影响输入数据。

### Q5: 如何只处理部分类别？

A: 将不需要处理的类别目录临时重命名（例如添加 `.bak` 后缀），或者移到其他位置。程序只会处理包含 `pre_images/` 和 `pre_labels/` 的目录。

### Q6: 如何调整训练参数？

A: 编辑 `config/config.yaml` 文件，或创建自定义配置文件并通过 `--config` 参数指定。

## 高级用法

### 使用不同的训练配置

```bash
# 创建自定义配置
cp config/config.yaml /data/my_config.yaml
# 编辑 my_config.yaml，调整 epochs, batch_size 等参数

# 使用自定义配置训练
python scripts/train_by_category.py \
    --data-root /data/my_dataset \
    --config /data/my_config.yaml
```

### 处理不同位置的多个数据集

```bash
# 创建批处理脚本 (Windows: batch_train.bat)
@echo off
python scripts/train_by_category.py --data-root D:\data\project1
python scripts/train_by_category.py --data-root D:\data\project2
python scripts/train_by_category.py --data-root D:\data\project3

# 或 (Linux/Mac: batch_train.sh)
#!/bin/bash
python scripts/train_by_category.py --data-root /data/project1
python scripts/train_by_category.py --data-root /data/project2
python scripts/train_by_category.py --data-root /data/project3
```

## 查看帮助

```bash
python scripts/train_by_category.py --help
```

## 日志文件

日志保存在：`logs/train_by_category.log`

查看日志以了解详细的训练过程和可能的错误信息。

## 实际应用场景

### 场景1：工业设备检测项目

```
/projects/factory_inspection/
├── pointer_meters/      # 指针表：50张已标注 + 2000张待标注
├── digital_displays/    # 数字显示：30张已标注 + 1500张待标注
├── warning_lights/      # 警示灯：40张已标注 + 1000张待标注
└── valve_positions/     # 阀门位置：25张已标注 + 800张待标注

# 一键处理所有设备类型
python scripts/train_by_category.py --data-root /projects/factory_inspection
```

### 场景2：多客户项目管理

```
/clients/
├── client_A/
│   ├── product1/
│   └── product2/
├── client_B/
│   ├── component1/
│   └── component2/
└── client_C/
    └── part1/

# 分别处理每个客户的数据
python scripts/train_by_category.py --data-root /clients/client_A
python scripts/train_by_category.py --data-root /clients/client_B
python scripts/train_by_category.py --data-root /clients/client_C
```
