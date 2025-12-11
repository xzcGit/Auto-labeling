"""Batch training script for multiple categories"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, save_config, setup_logger, ensure_dir
from src.data_processor import DatasetOrganizer
from src.trainer import YOLOTrainer
from src.auto_annotator import AutoAnnotator


def process_category(category_name: str, category_path: Path, base_config: Dict[str, Any], logger) -> bool:
    """Process a single category: prepare data, train model, and auto-annotate if needed
    
    Args:
        category_name: Name of the category
        category_path: Path to the category's raw data
        base_config: Base configuration dictionary
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category_name}")
        logger.info(f"{'='*60}")
        
        # Create category-specific paths
        category_data_root = Path(f"data/{category_name}")
        category_model_dir = Path(f"models/trained/{category_name}")
        category_output_dir = Path(f"output/{category_name}_predictions")
        category_dataset_config = Path(f"config/{category_name}_dataset.yaml")
        
        # Create category-specific config
        category_config = base_config.copy()
        category_config['paths']['data_root'] = str(category_data_root)
        category_config['paths']['model_root'] = str(category_model_dir.parent)
        category_config['paths']['output_root'] = str(category_output_dir)
        
        # Step 1: Prepare dataset
        logger.info(f"[{category_name}] Step 1: Preparing dataset...")
        organizer = DatasetOrganizer(str(category_data_root))
        
        train_count, val_count = organizer.split_dataset(
            str(category_path),
            split_ratio=category_config['validation']['split_ratio'],
            shuffle=category_config['validation']['shuffle'],
            seed=category_config['validation']['random_seed']
        )
        logger.info(f"[{category_name}] Dataset prepared: {train_count} train, {val_count} val")
        
        # Update dataset config path
        dataset_config_path = category_data_root.parent / "config" / "dataset_config.yaml"
        if dataset_config_path.exists():
            dataset_config_path.rename(category_dataset_config)
            logger.info(f"[{category_name}] Dataset config saved to {category_dataset_config}")
        
        # Step 2: Train model
        logger.info(f"[{category_name}] Step 2: Training model...")
        trainer = YOLOTrainer(category_config)
        trainer.load_model()
        
        # Override project path for this category
        ensure_dir(category_model_dir)
        results = trainer.train(str(category_dataset_config))
        logger.info(f"[{category_name}] Training completed")
        
        # Step 3: Auto-annotate unlabeled data if exists
        unlabeled_dir = category_path.parent / f"{category_name}_unlabeled" / "images"
        if unlabeled_dir.exists():
            logger.info(f"[{category_name}] Step 3: Auto-annotating unlabeled data...")
            
            # Find best model weights
            weights_dir = category_model_dir / "train" / "weights"
            best_weights = weights_dir / "best.pt"
            
            if best_weights.exists():
                annotator = AutoAnnotator(str(best_weights), category_config)
                ensure_dir(category_output_dir)
                stats = annotator.annotate_images(str(unlabeled_dir), str(category_output_dir))
                logger.info(f"[{category_name}] Auto-annotation completed: {stats}")
            else:
                logger.warning(f"[{category_name}] Best weights not found at {best_weights}")
        else:
            logger.info(f"[{category_name}] No unlabeled data found, skipping auto-annotation")
        
        logger.info(f"[{category_name}] ✓ Category processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"[{category_name}] ✗ Error processing category: {str(e)}", exc_info=True)
        return False


def main():
    """Main function to process all categories"""
    # Setup logger
    logger = setup_logger(__name__, "logs/train_by_category.log")
    logger.info("Starting batch training by category")
    
    # Load base configuration
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    base_config = load_config(config_path)
    logger.info(f"Loaded base configuration from {config_path}")
    
    # Scan for categories in data/raw/
    raw_data_dir = Path("data/raw")
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        return
    
    # Find all category directories (must contain images/ and labels/ subdirectories)
    categories = []
    for item in raw_data_dir.iterdir():
        if item.is_dir() and not item.name.endswith("_unlabeled"):
            images_dir = item / "images"
            labels_dir = item / "labels"
            if images_dir.exists() and labels_dir.exists():
                categories.append((item.name, item))
    
    if not categories:
        logger.warning(f"No valid categories found in {raw_data_dir}")
        logger.info("Each category should have 'images/' and 'labels/' subdirectories")
        return
    
    logger.info(f"Found {len(categories)} categories to process: {[c[0] for c in categories]}")
    
    # Process each category
    results = {}
    for category_name, category_path in categories:
        success = process_category(category_name, category_path, base_config, logger)
        results[category_name] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Batch Training Summary")
    logger.info(f"{'='*60}")
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    logger.info(f"Total categories: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    for category, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {category}")
    
    logger.info("Batch training completed")


if __name__ == "__main__":
    main()