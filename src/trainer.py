"""Model training module"""

from ultralytics import YOLO
from pathlib import Path
from .utils import setup_logger, ensure_dir


class YOLOTrainer:
    """YOLO model trainer"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(__name__)
        self.model = None
        
    def load_model(self):
        """Load YOLO model"""
        model_type = self.config['training']['model_type']
        model_size = self.config['training']['model_size']
        pretrained = self.config['training']['pretrained']
        
        model_name = f"{model_type}{model_size}"
        if pretrained:
            model_name += ".pt"
        
        self.logger.info(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        return self.model
    
    def train(self, data_config: str):
        """Train the model"""
        if self.model is None:
            self.load_model()
        
        train_cfg = self.config['training']
        output_dir = Path(self.config['paths']['model_root']) / "trained"
        ensure_dir(output_dir)
        
        self.logger.info("Starting training...")
        results = self.model.train(
            data=data_config,
            epochs=train_cfg['epochs'],
            batch=train_cfg['batch_size'],
            imgsz=train_cfg['img_size'],
            device=train_cfg['device'],
            workers=train_cfg['workers'],
            patience=train_cfg['patience'],
            save_period=train_cfg.get('save_period', 10),
            project=str(output_dir),
            name='train',
            exist_ok=True
        )
        
        self.logger.info("Training completed")
        return results
    
    def validate(self):
        """Validate the model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        self.logger.info("Running validation...")
        metrics = self.model.val()
        return metrics