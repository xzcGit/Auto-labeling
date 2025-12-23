# YOLO 图像自动标注系统

基于 YOLO 的图像自动标注工具，用少量标注数据训练模型，批量标注大规模数据集。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 数据准备

将标注数据放在 `data/raw/` 下，YOLO 格式（`class_id x_center y_center width height`）：

```
data/raw/
├── images/
│   └── *.jpg
└── labels/
    └── *.txt
```

### 一键训练

**默认模式**（扫描 `data/raw/` 目录）：

```bash
python scripts/train_by_category.py
```

**自定义路径模式**：

```bash
python scripts/train_by_category.py --data-root /path/to/data
```

自动扫描指定目录下所有类别，训练模型并标注待标注数据。详见[多类别批量训练](#多类别批量训练)。

## 分步执行

```bash
# 1. 准备数据集
python scripts/prepare_data.py --data-dir data/raw --output-dir data

# 2. 训练模型
python scripts/train_model.py --config config/config.yaml --data config/dataset_config.yaml

# 3. 自动标注
python scripts/auto_label.py --model models/trained/train/weights/best.pt --images data/unlabeled/images --output output/predictions
```

## 多类别批量训练

支持两种模式：

### 模式一：默认模式（推荐）

使用 `data/raw/` 作为根目录，数据结构：

```
data/raw/
├── category_a/           # 已标注数据
│   ├── images/
│   └── labels/
├── category_a_unlabeled/ # 待标注数据（可选）
│   └── images/
└── category_b/
    ├── images/
    └── labels/
```

运行命令：

```bash
python scripts/train_by_category.py
```

### 模式二：自定义路径模式

使用自定义路径作为根目录，数据结构：

```
/path/to/data/
├── category_a/
│   ├── pre_images/      # 已标注数据
│   ├── pre_labels/      # 已标注数据
│   ├── images/          # 待标注数据（可选）
│   ├── category/        # 输出：训练/验证集划分（自动生成）
│   ├── models/          # 输出：训练好的模型（自动生成）
│   └── labels/          # 输出：自动标注结果（自动生成）
└── category_b/
    ├── pre_images/
    ├── pre_labels/
    └── images/
```

运行命令：

```bash
python scripts/train_by_category.py --data-root /path/to/data
```

## 配置

编辑 `config/config.yaml`：

```yaml
training:
  model_type: "yolov8"    # yolov5, yolov8, yolov11
  model_size: "n"         # n, s, m, l, x
  epochs: 300
  batch_size: 4
  device: "cuda"

auto_annotation:
  confidence_threshold: 0.25
```

## 输出

标注结果按置信度分级：

```
output/predictions/labels/
├── high_conf/      # >0.7 可直接使用
├── medium_conf/    # 0.5-0.7 建议抽查
└── low_conf/       # <0.5 需人工复审
```

## 项目结构

```
├── config/          # 配置文件
├── data/            # 数据集
├── models/          # 模型权重
├── output/          # 输出结果
├── src/             # 核心代码
├── scripts/         # 命令行脚本
└── myutils/         # 辅助工具
```

## License

MIT