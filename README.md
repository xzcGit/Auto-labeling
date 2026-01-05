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

优先推荐使用共享模型目录（默认：`models/shared/`）：
- 训练输出：`models/shared/<category>/train/weights/best.pt`
- 轻量注册表：`models/model_registry.yaml`（类别 → best.pt）

关键参数：
- `--shared-model-root "models/shared"`：不同场站同名类别共享权重
- `--train-init reuse`：需要训练时，用已解析到的同类别权重热启动
- `--model-map "config/model_map.example.yaml"`：显式指定某些类别权重
- `--pretrained-root "/path/to/weights_dir"` / `--pretrained-model "/path/to/best.pt"`：直接用已有权重做标注

## 标注输出

两种布局：
- `triage`：按置信度分文件夹（`output/.../labels/high_conf|medium_conf|low_conf`）
- `yolo`：直接写 `labels/*.txt`（适合“数据目录即项目目录”的增量标注；并生成 `labels/_auto_label_report.json`）

## 配置

主要配置在 `config/config.yaml`（训练/推理阈值、设备、batch 等）。

## License

MIT
