# Auto-Labeling-Codex 使用说明

> 基于 YOLO 的自动标注工具，支持模型训练、自动标注和跨场站模型复用

---

## 一、项目概述

### 核心功能
- **模型训练**：用少量人工标注数据训练 YOLO 检测模型
- **自动标注**：用训练好的模型对未标注图片进行自动标注
- **模型复用**：同类别跨场站共享权重，避免重复训练
- **增量标注**：自动跳过已标注图片，仅处理新增数据

### 核心脚本
| 脚本 | 用途 |
|------|------|
| `scripts/train_by_station.py` | **主入口**：按场站批处理（推荐） |
| `scripts/train_by_category.py` | 按类别批处理（适用于 data/raw 目录结构） |

---

## 二、环境准备

### 步骤1：安装依赖
```bash
cd F:\code\utils\auto-labeling-codex
pip install -r requirements.txt
```

### 步骤2：检查 GPU（可选但推荐）
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
若为 `True` 表示 GPU 可用，训练速度会快很多。

---

## 三、数据目录结构

### 场站模式目录结构（推荐）

```
<stations_root>/                    # 如: F:\code\utils\19-metertools
├── <场站1>/                        # 如: 巴里坤1
│   ├── <类别1>/                    # 如: door
│   │   ├── pre_images/             # 已标注图片（用于训练）
│   │   ├── pre_labels/             # 已标注标签（YOLO txt格式）
│   │   ├── images/                 # 待标注图片
│   │   └── labels/                 # 自动标注输出目录
│   └── <类别2>/
│       └── *.jpeg                  # 平铺图片模式（flat_images）
├── <场站2>/
│   └── det/                        # 可选的 det 子目录
│       └── <类别>/
└── ...
```

### 支持的三种布局

| 布局类型 | 目录特征 | 说明 |
|---------|---------|------|
| **pre_labeled** | 有 `pre_images/` + `pre_labels/` | 有标注数据，可训练+标注 |
| **dir_images** | 有 `images/` 子目录 | 有待标注图片 |
| **flat_images** | 图片直接在类别目录下 | 仅图片，需已有权重才能标注 |

### YOLO 标签格式
```
# 每行格式: class_id x_center y_center width height
# 所有坐标为归一化值 (0-1)
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.25
```

---

## 四、核心命令

### 基本命令格式
```bash
python scripts/train_by_station.py --stations-root "<数据根目录>" [选项]
```

### 常用选项一览

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--stations-root` | 数据根目录（**必填**） | - |
| `--station` | 指定场站名（可多次使用） | 处理全部 |
| `--category` | 指定类别名（可多次使用） | 处理全部 |
| `--action` | 操作类型 | `annotate` |
| `--train-init` | 训练初始化策略 | `reuse` |
| `--force-train` | 强制重新训练 | False |
| `--output-layout` | 输出布局 | `yolo` |
| `--no-skip-existing` | 不跳过已标注图片 | False |
| `--shared-model-root` | 共享模型目录 | `models/shared` |
| `--registry` | 模型注册表路径 | `models/model_registry.yaml` |

### action 参数详解

| 值 | 说明 |
|----|------|
| `annotate` | 仅标注（需已有权重） |
| `train` | 仅训练（需 pre_images + pre_labels） |
| `train_and_annotate` | 训练后标注 |

### train-init 参数详解

| 值 | 说明 |
|----|------|
| `base` | 从零开始训练（使用预训练的 YOLOv8 基础权重） |
| `reuse` | 热启动训练（复用已有同类别权重，提升小样本效果） |

---

## 五、使用场景示例

### 场景A：用已有模型标注新图片

**前提**：该类别已有训练好的权重

```bash
# 标注所有场站的 pointer 类别
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --category "pointer"
```

**预期输出**：
```
📁 各场站/pointer/labels/
├── image001.txt          # YOLO格式标签文件
├── image002.txt
├── ...
└── _auto_label_report.json  # 统计报告
```

**控制台输出示例**：
```
[INFO] 扫描场站根目录: F:\code\utils\19-metertools
[INFO] 发现 5 个场站，1 个类别
[INFO] 处理: 巴里坤1/pointer
[INFO]   权重来源: trained (models/shared/pointer/train/weights/best.pt)
[INFO]   待标注图片: 120 张，跳过已标注: 0 张
[INFO]   标注完成: 高置信度 95, 中置信度 20, 低置信度 5
[INFO] 处理完成，共处理 5 个场站
```

### 场景B：训练新类别并标注

**前提**：有 `pre_images` + `pre_labels` 预标注数据

```bash
# 训练并标注 door 类别
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --station "汇总" ^
  --category "door" ^
  --action train_and_annotate
```

**预期输出**：
```
📁 models/shared/door/train/
├── weights/
│   ├── best.pt           # 最佳权重（验证集表现最好）
│   └── last.pt           # 最后一轮权重
├── results.csv           # 训练指标记录
├── confusion_matrix.png  # 混淆矩阵
└── results.png           # 训练曲线图

📁 汇总/door/labels/
├── *.txt                 # 自动生成的标签文件
└── _auto_label_report.json

📄 models/model_registry.yaml（自动更新）
  door: models/shared/door/train/weights/best.pt
```

**控制台输出示例**：
```
[INFO] 处理: 汇总/door
[INFO]   检测到训练数据: pre_images=30, pre_labels=30
[INFO]   开始训练...
[INFO]   Epoch 1/300: mAP50=0.45, mAP50-95=0.28
[INFO]   ...
[INFO]   Epoch 150/300: mAP50=0.92, mAP50-95=0.71 (early stop)
[INFO]   训练完成，权重保存至: models/shared/door/train/weights/best.pt
[INFO]   开始标注: 200 张图片
[INFO]   标注完成: 高置信度 180, 中置信度 15, 低置信度 5
```

### 场景C：处理特定场站的所有类别

```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --station "巴里坤1"
```

**预期输出**：
```
📁 巴里坤1/
├── door/labels/          # door类别的标签
│   ├── *.txt
│   └── _auto_label_report.json
├── light/labels/         # light类别的标签
│   ├── *.txt
│   └── _auto_label_report.json
└── pointer/labels/       # pointer类别的标签
    ├── *.txt
    └── _auto_label_report.json
```

**控制台输出示例**：
```
[INFO] 扫描场站根目录: F:\code\utils\19-metertools
[INFO] 过滤场站: ['巴里坤1']
[INFO] 发现 1 个场站，3 个类别
[INFO] 处理: 巴里坤1/door ... 完成 (50张)
[INFO] 处理: 巴里坤1/light ... 完成 (80张)
[INFO] 处理: 巴里坤1/pointer ... 完成 (120张)
[INFO] 处理完成，共处理 1 个场站，3 个类别
```

### 场景D：处理多个特定类别

```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --category "door" ^
  --category "light" ^
  --category "pointer"
```

**预期输出**：
```
📁 各场站/
├── <场站1>/door/labels/
├── <场站1>/light/labels/
├── <场站1>/pointer/labels/
├── <场站2>/door/labels/
├── ...
```

**控制台输出示例**：
```
[INFO] 扫描场站根目录: F:\code\utils\19-metertools
[INFO] 过滤类别: ['door', 'light', 'pointer']
[INFO] 发现 5 个场站，3 个类别
[INFO] 处理: 巴里坤1/door ... 完成
[INFO] 处理: 巴里坤1/light ... 完成
[INFO] 处理: 巴里坤1/pointer ... 完成
[INFO] 处理: 汇总/door ... 完成
...
[INFO] 处理完成，共处理 5 个场站，3 个类别
```

### 场景E：全量处理（首次运行）

```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --action train_and_annotate
```

**预期输出**：
```
📁 models/shared/
├── door/train/weights/best.pt      # 各类别训练的权重
├── light/train/weights/best.pt
├── pointer/train/weights/best.pt
└── ...

📁 各场站/各类别/labels/
├── *.txt                           # 所有图片的标签
└── _auto_label_report.json

📄 models/model_registry.yaml       # 注册表记录所有权重
```

**控制台输出示例**：
```
[INFO] 扫描场站根目录: F:\code\utils\19-metertools
[INFO] 发现 10 个场站，5 个类别
[INFO] 使用模式: train_and_annotate
[INFO] 处理: 汇总/door (有训练数据，开始训练...)
[INFO]   训练完成，mAP50=0.89
[INFO]   标注完成: 200 张
[INFO] 处理: 汇总/light (无训练数据，降级为标注)
[INFO]   使用权重: models/shared/light/train/weights/best.pt
[INFO]   标注完成: 150 张
...
[INFO] 全部处理完成
[INFO] 统计: 训练 3 个类别，标注 10 个场站，共 2500 张图片
```

### 场景F：增量标注（新增图片后）

```bash
# 默认跳过已有 labels/*.txt 的图片
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools"
```

**预期输出**：
```
📁 <类别>/labels/
├── old_image001.txt      # 已存在，跳过
├── old_image002.txt      # 已存在，跳过
├── new_image001.txt      # 新增标签 ✓
├── new_image002.txt      # 新增标签 ✓
└── _auto_label_report.json  # 更新统计
```

**控制台输出示例**：
```
[INFO] 处理: 巴里坤1/pointer
[INFO]   权重来源: trained
[INFO]   待标注图片: 150 张，跳过已标注: 120 张
[INFO]   实际处理: 30 张新增图片
[INFO]   标注完成: 高置信度 25, 中置信度 4, 低置信度 1
```

### 场景G：强制重新标注所有图片

```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --no-skip-existing
```

**预期输出**：
```
📁 <类别>/labels/
├── image001.txt          # 覆盖更新
├── image002.txt          # 覆盖更新
├── image003.txt          # 覆盖更新
└── _auto_label_report.json  # 重新统计
```

**控制台输出示例**：
```
[INFO] 处理: 巴里坤1/pointer
[INFO]   权重来源: trained
[INFO]   强制模式: 不跳过已标注图片
[INFO]   待标注图片: 150 张，跳过: 0 张
[INFO]   标注完成: 高置信度 120, 中置信度 25, 低置信度 5
[WARNING] 已覆盖 120 个已存在的标签文件
```

### 场景H：强制重新训练模型

```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --action train_and_annotate ^
  --force-train
```

**预期输出**：
```
📁 models/shared/<category>/train/
├── weights/
│   ├── best.pt           # 新训练的权重（覆盖旧权重）
│   └── last.pt
├── results.csv           # 新的训练记录
└── results.png           # 新的训练曲线

📁 models/shared/<category>/train2/   # 旧权重自动备份（如果存在）
└── weights/best.pt

📄 models/model_registry.yaml         # 更新为新权重路径
```

**控制台输出示例**：
```
[INFO] 处理: 汇总/door
[INFO]   检测到已有权重: models/shared/door/train/weights/best.pt
[INFO]   强制训练模式: 忽略已有权重，重新训练
[INFO]   开始训练（从基础权重）...
[INFO]   Epoch 1/300: mAP50=0.42
[INFO]   ...
[INFO]   Epoch 180/300: mAP50=0.94 (early stop)
[INFO]   新权重保存至: models/shared/door/train/weights/best.pt
[INFO]   注册表已更新
[INFO]   开始标注...
```

---

## 六、模型复用机制

### 权重查找优先级（默认模式）

1. `trained` - 本地训练权重 (`models/shared/<category>/train/weights/best.pt`)
2. `registry` - 注册表记录 (`model_registry.yaml`)
3. `model_map` - 显式映射文件
4. `pretrained_root` - 预训练目录
5. `pretrained_model` - 单个预训练文件

### 使用预训练权重的方式

#### 方式1：共享模型目录（推荐）
```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --shared-model-root "models/shared"
```

#### 方式2：模型注册表
```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --registry "models/model_registry.yaml"
```

#### 方式3：显式模型映射文件
创建 `config/model_map.yaml`：
```yaml
door: "path/to/door_weights.pt"
light: "models/shared/light/train/weights/best.pt"
pointer: "models/trained/pointer/train/weights/best.pt"
```

```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --model-map "config/model_map.yaml"
```

#### 方式4：预训练权重目录
```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --pretrained-root "path/to/pretrained_weights"
```

支持的目录结构：
- `<pretrained_root>/<category>.pt`
- `<pretrained_root>/<category>/best.pt`
- `<pretrained_root>/<category>/train/weights/best.pt`

#### 方式5：单个预训练模型（所有类别共用）
```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --pretrained-model "path/to/yolov8n.pt"
```

### 优先使用预训练权重
```bash
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --pretrained-root "path/to/pretrained" ^
  --prefer-pretrained
```

---

## 七、输出说明

### YOLO 布局（默认）
```
<类别>/labels/
├── image1.txt
├── image2.txt
├── ...
└── _auto_label_report.json    # 统计报告
```

### Triage 布局
```bash
python scripts/train_by_station.py ... --output-layout triage
```
```
<类别>/labels/
├── high_conf/      # 高置信度标签
├── medium_conf/    # 中置信度标签
└── low_conf/       # 低置信度标签（需人工复核）
```

### 统计报告格式 (`_auto_label_report.json`)
```json
{
  "total": 100,
  "high_conf": 75,
  "medium_conf": 20,
  "low_conf": 5
}
```

---

## 八、配置文件

### 主配置文件：`config/config.yaml`

```yaml
project:
  name: "auto_annotation_project"
  version: "1.0.0"

paths:
  data_root: "./data"
  model_root: "./models"
  output_root: "./output"

training:
  model_type: "yolov8"        # yolov5, yolov8, yolov11
  model_size: "n"             # n(最小), s, m, l, x(最大)
  pretrained: true
  epochs: 300                 # 训练轮数
  batch_size: 4               # 批次大小（显存不足时减小）
  img_size: 640               # 图像大小
  device: "cuda"              # cuda, cpu, mps
  workers: 2
  patience: 100               # 早停耐心值
  amp: true                   # 混合精度训练
  freeze: 10                  # 冻结前N层（小样本推荐）

  # 优化器参数
  lr0: 0.001                  # 初始学习率
  lrf: 0.01                   # 最终学习率因子
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 5

  # 数据增强
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 5.0
  scale: 0.9
  copy_paste: 0.3

validation:
  split_ratio: 0.1            # 验证集比例
  shuffle: true
  random_seed: 42

auto_annotation:
  confidence_threshold: 0.4   # 检测置信度阈值
  iou_threshold: 0.45         # NMS IOU阈值
  max_det: 300                # 最大检测数
  review_threshold: 0.5       # 高/低置信度分界
  batch_size: 1               # 推理批次（显存小时用1）
  img_size: 640
  half: true                  # FP16推理
  chunk_size: 50              # 分块处理大小
```

### 模型注册表：`models/model_registry.yaml`
```yaml
# category_name: weights_path
pointer: F:\code\utils\auto-labeling-codex\models\trained\pointer\train\weights\best.pt
door: models/shared/door/train/weights/best.pt
```

---

## 九、智能降级机制

### 训练降级为标注
当执行 `train` 或 `train_and_annotate` 但类别没有 `pre_images` + `pre_labels` 时：
- 自动降级为 `annotate`（若有可用权重）
- 若也无权重，则跳过该类别

### 标注自动触发训练
当执行 `annotate` 但没有可用权重时：
- 若存在可训练数据（`pre_images` + `pre_labels` 或 `images` + `labels`）
- 自动训练一次再继续标注

---

## 十、推荐工作流

### 首次使用流程

```bash
# Step 1: 测试已有模型
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --station "测试" ^
  --category "pointer"

# Step 2: 检查输出
# 查看 测试/pointer/labels/ 目录

# Step 3: 训练新类别（如 door）
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --station "汇总" ^
  --category "door" ^
  --action train_and_annotate

# Step 4: 用训练好的模型标注其他场站
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --category "door"
```

### 日常增量标注

```bash
# 新增图片后重新运行（自动跳过已标注）
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools"
```

### 模型迭代优化

```bash
# 基于已有权重热启动训练（小样本效果更好）
python scripts/train_by_station.py ^
  --stations-root "F:\code\utils\19-metertools" ^
  --action train_and_annotate ^
  --train-init reuse ^
  --force-train
```

---

## 十一、准备新类别训练数据

### 步骤1：创建目录结构
```
<场站>/<类别>/
├── pre_images/    # 放少量已标注图片（建议10-50张）
├── pre_labels/    # 对应的 YOLO 格式标签
└── images/        # 待自动标注的图片
```

### 步骤2：标注工具推荐

| 工具 | 安装/使用 |
|------|----------|
| LabelImg | `pip install labelImg && labelImg` |
| CVAT | 在线标注平台 |
| Label Studio | `pip install label-studio && label-studio` |
| Roboflow | 在线标注+数据增强 |

### 步骤3：标注建议
- 每个类别至少 10-20 张标注图片
- 覆盖不同场景、光照、角度
- 标注框要准确贴合目标边界

---

## 十二、常见问题

| 问题 | 解决方案 |
|------|----------|
| GPU 显存不足 (CUDA OOM) | 修改 `config.yaml`：`batch_size: 1`，`chunk_size: 20` |
| 没有权重无法标注 | 准备 pre_images+pre_labels，使用 `--action train_and_annotate` |
| 标注质量差 | 增加训练数据 / 调低 `confidence_threshold` / 增加 epochs |
| 训练太慢 | 减少 `epochs` / 使用 `model_size: "n"` / 使用 GPU |
| 找不到类别 | 检查目录名是否有空格/特殊字符，确保图片格式正确 |
| 权重不复用 | 检查 `--shared-model-root` 路径，或使用 `--registry` |

---

## 十三、日志与调试

### 日志位置
```
logs/train_by_station.log
```

### 查看详细输出
运行命令时会实时输出处理进度，包括：
- 扫描到的场站和类别
- 每个类别的处理状态
- 权重来源（trained/registry/model_map 等）
- 训练进度和指标
- 标注统计

---

## 十四、项目文件结构

```
auto-labeling-codex/
├── scripts/
│   ├── train_by_station.py    # 主入口（场站模式）
│   ├── train_by_category.py   # 类别模式入口
│   ├── auto_label.py          # 单独标注脚本
│   ├── train_model.py         # 单独训练脚本
│   └── prepare_data.py        # 数据准备脚本
├── src/
│   ├── station_scanner.py     # 场站扫描
│   ├── category_runner.py     # 类别处理器
│   ├── category_pipeline.py   # 处理流水线
│   ├── auto_annotator.py      # 自动标注器
│   ├── trainer.py             # YOLO训练器
│   ├── predictor.py           # YOLO预测器
│   ├── data_processor.py      # 数据处理
│   ├── model_registry.py      # 模型注册表
│   └── utils.py               # 工具函数
├── config/
│   ├── config.yaml            # 主配置
│   ├── model_map.example.yaml # 模型映射示例
│   └── dataset_config.yaml    # 数据集配置模板
├── models/
│   ├── shared/                # 共享模型目录
│   ├── trained/               # 已训练模型
│   └── model_registry.yaml    # 模型注册表
├── logs/                      # 日志目录
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

---

## 十五、命令速查表

```bash
# 标注单个类别
python scripts/train_by_station.py --stations-root "数据目录" --category "类别名"

# 训练并标注
python scripts/train_by_station.py --stations-root "数据目录" --action train_and_annotate

# 处理特定场站
python scripts/train_by_station.py --stations-root "数据目录" --station "场站名"

# 强制重新训练
python scripts/train_by_station.py --stations-root "数据目录" --action train_and_annotate --force-train

# 不跳过已标注
python scripts/train_by_station.py --stations-root "数据目录" --no-skip-existing

# 使用外部预训练权重
python scripts/train_by_station.py --stations-root "数据目录" --pretrained-root "权重目录" --prefer-pretrained

# 使用模型映射文件
python scripts/train_by_station.py --stations-root "数据目录" --model-map "config/model_map.yaml"

# 输出为置信度分级布局
python scripts/train_by_station.py --stations-root "数据目录" --output-layout triage
```

---

*文档生成时间: 2026-01-07*
