"""Utility functions for the auto-annotation system"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def get_image_files(directory: str, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Get all image files in directory"""
    image_files = set()
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))         # linux系统会区分大小写，但是windows不区分大小写
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(list(image_files))


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert bounding box to YOLO format (normalized)"""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def convert_yolo_to_bbox(yolo_bbox, img_width, img_height):
    """Convert YOLO format to bounding box coordinates"""
    x_center, y_center, width, height = yolo_bbox
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return x_min, y_min, x_max, y_max