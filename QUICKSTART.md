# 快速入门指南

本指南将帮助您在5分钟内开始使用自动标注系统。

## 📦 第一步：安装

### 1. 克隆或下载项目

```bash
cd f:/code/utils/model_train
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

安装完成后验证：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
```

## 📊 第二步：准备数据

### 数据格式要求

您需要准备少量已标注的图像（建议50-500张），格式如下：

```
data/raw/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── labels/
    ├── img_001.txt
    ├── img_002.txt
    └── ...
```

### YOLO标注格式

每个标注文件（.txt）的格式为：

```
class_id x_center y_center width height
```

- `class_id`: 类别编号（从0开始）
- `x_center, y_center`: 边界框中心点（归一化到0-1）
- `width, height`: 边界框宽高（归一化到0-1）

**示例**：
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### 将未标注图像放入

```
data/unlabeled/images/
├── new_img_001.jpg
├── new_img_002.jpg
└── ...
```

## 🚀 第三步：运行系统

### 方式一：一键运行（推荐新手）

```bash
python run_pipeline.py --mode full
```

这将自动完成：
1. ✅ 数据准备和划分
2. ✅ 模型训练
3. ✅ 自动标注

### 方式二：分步执行（推荐高级用户）

#### 步骤1：准备数据

```bash
python scripts/prepare_data.py --data-dir data/raw --output-dir data
```

输出：
```
✓ Dataset prepared successfully!
  Train samples: 80
  Val samples: 20
```

#### 步骤2：训练模型

```bash
python scripts/train_model.py --config config/config.yaml
```

这将需要一些时间（取决于数据量和硬件）。训练完成后，最佳模型保存在：
```
models/trained/train/weights/best.pt
```

#### 步骤3：自动标注

```bash
python scripts/auto_label.py ^
  --model models/trained/train/weights/best.pt ^
  --images data/unlabeled/images ^
  --output output/predictions
```

## 📈 第四步：检查结果

### 查看标注结果

标注结果按置信度分级保存：

```
output/predictions/labels/
├── high_conf/       # 高置信度 (>0.7) - 可直接使用
├── medium_conf/     # 中等置信度 (0.5-0.7) - 建议抽查
└── low_conf/        # 低置信度 (<0.5) - 需要人工复审
```

### 查看统计信息

```bash
type output\predictions\statistics.json
```

输出示例：
```json
{
  "total": 1000,
  "high_conf": 750,
  "medium_conf": 200,
  "low_conf": 50
}
```

## 🎯 使用建议

### 针对不同数据量的配置

#### 小数据集 (<200张)
编辑 `config/config.yaml`:
```yaml
training:
  model_type: "yolov8"
  model_size: "n"      # 使用nano模型
  epochs: 150
  batch_size: 16
```

#### 中等数据集 (200-1000张)
```yaml
training:
  model_type: "yolov8"
  model_size: "s"      # 使用small模型
  epochs: 100
  batch_size: 16
```

#### 大数据集 (>1000张)
```yaml
training:
  model_type: "yolov8"
  model_size: "m"      # 使用medium模型
  epochs: 100
  batch_size: 32
```

### 调整置信度阈值

如果自动标注质量不满意，可以调整阈值：

```yaml
auto_annotation:
  confidence_threshold: 0.7    # 提高到0.7，更保守
  review_threshold: 0.6        # 提高复审阈值
```

## 💡 实战示例

### 示例1：检测3类物体（人、车、自行车）

```bash
# 1. 准备50张标注好的图像
# data/raw/images/ - 50张图像
# data/raw/labels/ - 50个标注文件（类别: 0=人, 1=车, 2=自行车）

# 2. 准备5000张待标注图像
# data/unlabeled/images/ - 5000张图像

# 3. 运行完整流程
python run_pipeline.py --mode full

# 4. 查看结果
dir output\predictions\labels\high_conf
```

### 示例2：只训练不标注

```bash
# 仅准备数据和训练
python run_pipeline.py --mode prepare
python run_pipeline.py --mode train

# 之后可以单独运行标注
python run_pipeline.py --mode annotate --model models/trained/train/weights/best.pt
```

## 🔧 故障排除

### 问题1：CUDA不可用

**症状**：训练使用CPU，速度很慢

**解决**：
```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False，重新安装支持CUDA的PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题2：内存不足

**症状**：CUDA out of memory

**解决**：减小batch_size
```yaml
training:
  batch_size: 8  # 从16减到8
```

### 问题3：找不到图像

**症状**：Found 0 images

**解决**：检查目录结构和文件扩展名
```bash
# 列出图像
dir data\raw\images\*.jpg
dir data\raw\images\*.png
```

### 问题4：训练不收敛

**症状**：Loss不下降

**解决**：
1. 检查标注是否正确
2. 增加训练轮数
3. 调整学习率（在Ultralytics配置中）
4. 确保使用预训练权重

## 📚 下一步

- 阅读 [`README.md`](README.md) 了解详细功能
- 查看 [`ARCHITECTURE.md`](ARCHITECTURE.md) 了解系统架构
- 根据需要调整 `config/config.yaml` 配置
- 尝试不同的YOLO版本和模型大小

## 🎓 学习资源

- [Ultralytics YOLO文档](https://docs.ultralytics.com/)
- [YOLO标注格式说明](https://docs.ultralytics.com/datasets/)
- [PyTorch文档](https://pytorch.org/docs/)

---

**祝您使用愉快！如有问题，请查看README或提交Issue。**