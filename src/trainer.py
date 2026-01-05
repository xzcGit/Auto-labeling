"""Model training module"""

from ultralytics import YOLO
from pathlib import Path
from typing import Optional
from .utils import setup_logger, ensure_dir


class YOLOTrainer:
    """YOLO model trainer"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(__name__)
        self.model = None
        
    def load_model(self, model_path: Optional[str] = None):
        """Load YOLO model.

        Args:
            model_path: Optional path to a .pt weights file to initialize from.
                When omitted, uses training.model_type/model_size/pretrained to
                build the default base model name.
        """
        if model_path:
            self.logger.info(f"Loading model from weights: {model_path}")
            self.model = YOLO(model_path)
            return self.model

        model_type = self.config["training"]["model_type"]
        model_size = self.config["training"]["model_size"]
        pretrained = self.config["training"]["pretrained"]

        model_name = f"{model_type}{model_size}"
        model_name += ".pt" if pretrained else ".yaml"

        self.logger.info(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        return self.model
    
    def train(self, data_config: str):
        """Train the model"""
        if self.model is None:
            self.load_model()

        train_cfg = self.config['training']
        # 直接使用model_root作为输出目录，不再追加'trained'
        output_dir = Path(self.config['paths']['model_root'])
        ensure_dir(output_dir)

        # 构建训练参数字典
        train_params = {
            'data': data_config,
            'epochs': train_cfg['epochs'],
            'batch': train_cfg['batch_size'],
            'imgsz': train_cfg['img_size'],
            'device': train_cfg['device'],
            'workers': train_cfg['workers'],
            'amp': train_cfg['amp'],
            'patience': train_cfg['patience'],
            'save_period': train_cfg.get('save_period', -1),
            'project': str(output_dir),
            'name': 'train',
            'exist_ok': True,
        }

        # 添加优化器参数（如果配置中存在）
        optimizer_params = ['lr0', 'lrf', 'momentum', 'weight_decay',
                           'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr']
        for param in optimizer_params:
            if param in train_cfg:
                train_params[param] = train_cfg[param]

        # 添加冻结层参数（小样本训练关键）
        if 'freeze' in train_cfg:
            train_params['freeze'] = train_cfg['freeze']

        # 添加数据增强参数（如果配置中存在）
        augmentation_params = [
            'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
            'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste'
        ]
        for param in augmentation_params:
            if param in train_cfg:
                train_params[param] = train_cfg[param]

        self.logger.info("Starting training with parameters:")
        self.logger.info(f"  Epochs: {train_params['epochs']}")
        self.logger.info(f"  Batch size: {train_params['batch']}")
        self.logger.info(f"  Learning rate: {train_params.get('lr0', 'default')}")
        self.logger.info(f"  Freeze layers: {train_params.get('freeze', 0)} (0=no freeze, 10=freeze backbone)")
        self.logger.info(f"  Data augmentation: mosaic={train_params.get('mosaic', 'default')}, "
                        f"mixup={train_params.get('mixup', 'default')}, "
                        f"copy_paste={train_params.get('copy_paste', 'default')}")

        results = self.model.train(**train_params)

        self.logger.info("Training completed")
        return results
    
    def validate(self):
        """Validate the model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        self.logger.info("Running validation...")
        metrics = self.model.val()
        return metrics
