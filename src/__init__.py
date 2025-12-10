"""
Auto-Annotation System for YOLO
A complete pipeline for training YOLO models and auto-labeling images
"""

__version__ = "1.0.0"
__author__ = "Auto-Annotation System"

from .utils import setup_logger, load_config
from .data_processor import DatasetOrganizer
from .trainer import YOLOTrainer
from .predictor import YOLOPredictor
from .auto_annotator import AutoAnnotator

__all__ = [
    'setup_logger',
    'load_config',
    'DatasetOrganizer',
    'YOLOTrainer',
    'YOLOPredictor',
    'AutoAnnotator',
]