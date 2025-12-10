"""Data processing and organization module"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
from .utils import setup_logger, ensure_dir, get_image_files, save_config


class DatasetOrganizer:
    """Organize and prepare dataset for YOLO training"""
    
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.logger = setup_logger(__name__)
        
    def validate_dataset(self, raw_dir: str) -> Tuple[List[Path], List[Path]]:
        """Validate raw dataset structure"""
        raw_path = Path(raw_dir)
        images_dir = raw_path / "images"
        labels_dir = raw_path / "labels"
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        
        image_files = get_image_files(images_dir)
        label_files = list(labels_dir.glob("*.txt"))
        
        self.logger.info(f"Found {len(image_files)} images and {len(label_files)} labels")
        
        # Check matching
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        matched = image_stems & label_stems
        
        self.logger.info(f"Matched pairs: {len(matched)}")
        
        return image_files, label_files
    
    def split_dataset(self, raw_dir: str, split_ratio: float = 0.2, 
                     shuffle: bool = True, seed: int = 42):
        """Split dataset into train and validation sets"""
        image_files, label_files = self.validate_dataset(raw_dir)
        
        # Get matched pairs
        image_dict = {f.stem: f for f in image_files}
        label_dict = {f.stem: f for f in label_files}
        matched_stems = list(set(image_dict.keys()) & set(label_dict.keys()))
        
        if shuffle:
            random.seed(seed)
            random.shuffle(matched_stems)
        
        # Split
        split_idx = int(len(matched_stems) * (1 - split_ratio))
        train_stems = matched_stems[:split_idx]
        val_stems = matched_stems[split_idx:]
        
        self.logger.info(f"Train: {len(train_stems)}, Val: {len(val_stems)}")
        
        # Copy files
        self._copy_files(train_stems, image_dict, label_dict, "train")
        self._copy_files(val_stems, image_dict, label_dict, "val")
        
        # Create dataset config
        self._create_dataset_config(train_stems, label_dict)
        
        return len(train_stems), len(val_stems)
    
    def _copy_files(self, stems: List[str], image_dict: dict, 
                   label_dict: dict, split: str):
        """Copy files to train/val directories"""
        img_dst = self.data_root / split / "images"
        lbl_dst = self.data_root / split / "labels"
        ensure_dir(img_dst)
        ensure_dir(lbl_dst)
        
        for stem in stems:
            if stem in image_dict:
                shutil.copy2(image_dict[stem], img_dst / image_dict[stem].name)
            if stem in label_dict:
                shutil.copy2(label_dict[stem], lbl_dst / label_dict[stem].name)
    
    def _create_dataset_config(self, train_stems: List[str], label_dict: dict):
        """Create dataset.yaml for YOLO"""
        # Extract class names from labels
        class_ids = set()
        for stem in train_stems:
            if stem in label_dict:
                with open(label_dict[stem], 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
        
        num_classes = len(class_ids)
        class_names = {i: f"class{i}" for i in sorted(class_ids)}
        
        config = {
            'path': str(self.data_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': num_classes,
            'names': class_names
        }
        
        config_path = self.data_root.parent / "config" / "dataset_config.yaml"
        save_config(config, str(config_path))
        self.logger.info(f"Dataset config saved to {config_path}")