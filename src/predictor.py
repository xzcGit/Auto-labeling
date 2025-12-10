"""Model prediction module"""

from ultralytics import YOLO
from pathlib import Path
from typing import List
from .utils import setup_logger


class YOLOPredictor:
    """YOLO model predictor"""
    
    def __init__(self, model_path: str, config: dict):
        self.model_path = model_path
        self.config = config
        self.logger = setup_logger(__name__)
        self.model = None
        
    def load_model(self):
        """Load trained model"""
        self.logger.info(f"Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        return self.model
    
    def predict_batch(self, image_paths: List[Path], **kwargs):
        """Predict on batch of images"""
        if self.model is None:
            self.load_model()
        
        auto_cfg = self.config.get('auto_annotation', {})
        conf = kwargs.get('conf', auto_cfg.get('confidence_threshold', 0.6))
        iou = kwargs.get('iou', auto_cfg.get('iou_threshold', 0.45))
        max_det = kwargs.get('max_det', auto_cfg.get('max_det', 300))
        
        results = self.model.predict(
            source=image_paths,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False
        )
        
        return results
    
    def filter_by_confidence(self, results, threshold: float = 0.5):
        """Filter predictions by confidence threshold"""
        high_conf = []
        medium_conf = []
        low_conf = []
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                low_conf.append(result)
                continue
                
            max_conf = float(result.boxes.conf.max())
            if max_conf >= 0.7:
                high_conf.append(result)
            elif max_conf >= threshold:
                medium_conf.append(result)
            else:
                low_conf.append(result)
        
        return high_conf, medium_conf, low_conf