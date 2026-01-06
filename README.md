# YOLO 自动标注工具

用少量人工标注训练/复用 YOLO 权重，对新增图片做增量自动标注。

## 安装

```bash
pip install -r "requirements.txt"
```

## 数据格式（两种）

### 1) 按类别批处理（category 模式）

默认根目录：`data/raw/`

```
data/raw/
├── category_a/
│   ├── images/          # 已标注图片
│   └── labels/          # 已标注标签（YOLO txt）
├── category_a_unlabeled/
│   └── images/          # 待标注图片（可选）
└── category_b/...
```

也支持自定义 root（详见 `dataroot.md`），关键目录为：`pre_images/` + `pre_labels/` + `images/`。

### 2) 按场站批处理（station / metertools 模式）

根目录包含多个场站：
```
<stations_root>/
  <station>/
    det/<category>/...    # 常见
    <category>/...        # 也支持“平铺图片目录”
    # 也兼容：<station> 目录本身就是一个“类别目录”
    #   <station>/{pre_images,pre_labels,images,...}
```

## 常用命令

### A. 新增图片的增量标注（推荐：场站模式）

对 `--stations-root` 下所有场站做增量标注（默认跳过已有 `labels/*.txt` 的图片）：

```bash
python3 "scripts/train_by_station.py" --stations-root "/mnt/f/code/utils/19-metertools"
```

只处理某个场站 / 某些类别：

```bash
python3 "scripts/train_by_station.py" --stations-root "/mnt/f/code/utils/19-metertools" --station "巴里坤1"
python3 "scripts/train_by_station.py" --stations-root "/mnt/f/code/utils/19-metertools" --category "door" --category "light"
```

有新场站目录或新图片加入后，重复执行同一条命令即可。

### B. 训练 + 标注（可选）

仅对存在 `pre_images/` + `pre_labels/` 的类别训练；缺少标注数据的类别会自动降级为 annotate：

```bash
python3 "scripts/train_by_station.py" --stations-root "/mnt/f/code/utils/19-metertools" --action "train_and_annotate"
```

补充：如果你执行的是 `--action annotate`，但当前类别没有任何可用权重，且目录里存在可训练的标注数据（`pre_images/` + `pre_labels/` 或 `images/` + `labels/`），工具会自动训练一次以继续标注。

### C. 按类别批处理（category 模式）

```bash
python3 "scripts/train_by_category.py"
python3 "scripts/train_by_category.py" --data-root "/path/to/data"
```

## 模型复用（跨场站/跨批次）

### 预训练模型优先级

系统通过多层级优先级机制查找可用权重：

**默认模式**（`--train-init reuse`）：
1. `trained` - 本地训练的权重（`models/shared/<category>/train/weights/best.pt`）
2. `registry` - 注册表权重（`model_registry.yaml` 记录的路径）
3. `model_map` - 模型映射文件显式指定的权重
4. `pretrained_root` - 预训练根目录下的权重
5. `pretrained_model` - 单个预训练模型文件

**优先预训练模式**（`--prefer-pretrained`）：
1. `model_map` - 显式指定的权重（最高优先级）
2. `pretrained_root` - 预训练根目录
3. `pretrained_model` - 单个预训练模型
4. `registry` - 注册表权重
5. `trained` - 本地训练的权重

### 使用预训练模型的五种方式

#### 1. 共享模型目录（推荐）

```bash
python3 "scripts/train_by_station.py" --stations-root "/path" --shared-model-root "models/shared"
```

- 同名类别自动共享权重
- 训练输出到 `models/shared/<category>/train/weights/best.pt`

#### 2. 模型注册表

```bash
python3 "scripts/train_by_station.py" --stations-root "/path" --registry "models/model_registry.yaml"
```

- 轻量级的类别→权重映射
- 训练后自动更新注册表

#### 3. 显式模型映射（最高优先级）

```bash
python3 "scripts/train_by_station.py" --stations-root "/path" --model-map "config/model_map.yaml"
```

配置文件示例（`config/model_map.yaml`）：
```yaml
door: /path/to/door_weights.pt
light: models/shared/light/train/weights/best.pt
meter: ../pretrained/meter_v2.pt
```

#### 4. 预训练根目录

```bash
python3 "scripts/train_by_station.py" --stations-root "/path" --pretrained-root "/path/to/weights_dir"
```

支持以下目录结构：
- `<pretrained_root>/<category>.pt`
- `<pretrained_root>/<category>/best.pt`
- `<pretrained_root>/<category>/train/weights/best.pt`
- `<pretrained_root>/<category>/weights/best.pt`

#### 5. 单个预训练模型（所有类别共用）

```bash
python3 "scripts/train_by_station.py" --stations-root "/path" --pretrained-model "/path/to/best.pt"
```

### 模型迭代机制

#### 热启动训练

使用已有权重作为起点继续训练（提升小样本场景效果）：

```bash
python3 "scripts/train_by_station.py" \
  --stations-root "/path" \
  --action train_and_annotate \
  --train-init reuse \
  --force-train
```

- `--train-init reuse`：用已解析到的同类别权重热启动
- `--force-train`：强制重新训练，即使存在权重

#### 智能降级机制

- 如果执行 `train` 或 `train_and_annotate`，但当前类别没有 `pre_images/` + `pre_labels/`，自动降级为 `annotate`
- 如果执行 `annotate`，但没有可用权重，且存在标注数据（`pre_images/` + `pre_labels/` 或 `images/` + `labels/`），自动训练一次以继续标注

#### 增量标注

默认跳过已标注图片（已有 `labels/*.txt` 的图片），实现增量标注：

```bash
# 首次运行：标注所有图片
python3 "scripts/train_by_station.py" --stations-root "/path"

# 新增图片后再次运行：仅标注新增图片
python3 "scripts/train_by_station.py" --stations-root "/path"

# 强制重新标注所有图片
python3 "scripts/train_by_station.py" --stations-root "/path" --no-skip-existing
```

### 典型工作流

**场景1：新增图片的增量标注**

```bash
# 第一次：有预标注数据，训练 + 标注
python3 "scripts/train_by_station.py" --stations-root "/path" --action train_and_annotate

# 后续：新增图片，仅标注（复用已训练权重）
python3 "scripts/train_by_station.py" --stations-root "/path" --action annotate
```

**场景2：跨场站权重复用**

```bash
# 所有场站共享同类别权重
python3 "scripts/train_by_station.py" \
  --stations-root "/path" \
  --shared-model-root "models/shared" \
  --train-init reuse
```

**场景3：使用外部预训练权重**

```bash
# 优先使用外部预训练权重，降级使用本地训练权重
python3 "scripts/train_by_station.py" \
  --stations-root "/path" \
  --pretrained-root "/path/to/pretrained" \
  --prefer-pretrained
```

**场景4：模型热启动迭代训练**

```bash
# 基于已有权重继续训练（提升效果）
python3 "scripts/train_by_station.py" \
  --stations-root "/path" \
  --action train_and_annotate \
  --train-init reuse \
  --force-train
```

## 标注输出

两种布局：
- `triage`：按置信度分文件夹（`output/.../labels/high_conf|medium_conf|low_conf`）
- `yolo`：直接写 `labels/*.txt`（适合“数据目录即项目目录”的增量标注；并生成 `labels/_auto_label_report.json`）

## 配置

主要配置在 `config/config.yaml`（训练/推理阈值、设备、batch 等）。

## License

MIT
