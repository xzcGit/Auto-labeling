"""Auto-annotation module for generating YOLO labels"""

import json
from pathlib import Path
from typing import List
from tqdm import tqdm
from .utils import setup_logger, ensure_dir, get_image_files
from .predictor import YOLOPredictor


class AutoAnnotator:
    """Auto-annotate images using trained YOLO model"""
    
    def __init__(self, model_path: str, config: dict):
        self.predictor = YOLOPredictor(model_path, config)
        self.config = config
        self.logger = setup_logger(__name__)
        
    def annotate_images(self, image_dir: str, output_dir: str):
        """Annotate all images in directory"""
        image_files = get_image_files(image_dir)
        self.logger.info(f"Found {len(image_files)} images to annotate")
        
        # Create output directories
        labels_dir = Path(output_dir) / "labels"
        high_dir = labels_dir / "high_conf"
        medium_dir = labels_dir / "medium_conf"
        low_dir = labels_dir / "low_conf"
        
        for d in [high_dir, medium_dir, low_dir]:
            ensure_dir(d)
        
        # Predict
        results = self.predictor.predict_batch(image_files)
        
        # Filter by confidence
        review_threshold = self.config['auto_annotation']['review_threshold']
        high_conf, medium_conf, low_conf = self.predictor.filter_by_confidence(
            results, review_threshold
        )
        
        # Generate labels
        stats = {
            'total': len(results),
            'high_conf': len(high_conf),
            'medium_conf': len(medium_conf),
            'low_conf': len(low_conf)
        }
        
        self._save_labels(high_conf, high_dir)
        self._save_labels(medium_conf, medium_dir)
        self._save_labels(low_conf, low_dir)
        
        # Save statistics
        stats_file = Path(output_dir) / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Annotation complete: {stats}")
        return stats
    
    def _save_labels(self, results, output_dir: Path):
        """Save YOLO format labels"""
        for result in tqdm(results, desc=f"Saving to {output_dir.name}"):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            img_path = Path(result.path)
            label_file = output_dir / f"{img_path.stem}.txt"
            
            with open(label_file, 'w') as f:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    xywhn = box.xywhn[0].tolist()
                    f.write(f"{cls} {' '.join(map(str, xywhn))}\n")