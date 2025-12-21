"""Batch training script for multiple categories
Supports two modes:
1. Default mode: Scan and process all categories in data/raw/
2. Custom path mode: Scan and process all categories in a custom root path
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, save_config, setup_logger, ensure_dir
from src.data_processor import DatasetOrganizer
from src.trainer import YOLOTrainer
from src.auto_annotator import AutoAnnotator


def process_category(
    category_name: str,
    category_root: Path,
    base_config: Dict[str, Any],
    logger,
    use_pre_prefix: bool = False
) -> bool:
    """Process a single category: prepare data, train model, and auto-annotate if needed

    Args:
        category_name: Name of the category
        category_root: Root path of the category (contains images/labels or pre_images/pre_labels)
        base_config: Base configuration dictionary
        logger: Logger instance
        use_pre_prefix: If True, look for pre_images/pre_labels instead of images/labels

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Category root: {category_root}")

        # Determine input directory names based on mode
        if use_pre_prefix:
            input_images_dir = "pre_images"
            input_labels_dir = "pre_labels"
            unlabeled_images_dir = "images"
        else:
            input_images_dir = "images"
            input_labels_dir = "labels"
            unlabeled_images_dir = f"{category_name}_unlabeled/images"

        # Validate input directories
        raw_images_path = category_root / input_images_dir
        raw_labels_path = category_root / input_labels_dir

        if not raw_images_path.exists():
            logger.error(f"[{category_name}] Input images directory not found: {raw_images_path}")
            return False

        if not raw_labels_path.exists():
            logger.error(f"[{category_name}] Input labels directory not found: {raw_labels_path}")
            return False

        # Setup output paths within the category root
        category_data_root = category_root / "category"
        category_model_dir = category_root / "models"
        category_output_dir = category_root / "labels"
        category_dataset_config = category_root / "dataset_config.yaml"

        # For unlabeled images
        if use_pre_prefix:
            unlabeled_dir = category_root / unlabeled_images_dir
        else:
            unlabeled_dir = category_root.parent / unlabeled_images_dir

        # Create category-specific config
        category_config = base_config.copy()
        category_config['paths']['data_root'] = str(category_data_root)
        category_config['paths']['model_root'] = str(category_model_dir)
        category_config['paths']['output_root'] = str(category_output_dir)

        # Step 1: Prepare dataset
        logger.info(f"[{category_name}] Step 1: Preparing dataset...")
        logger.info(f"[{category_name}] Reading from: {raw_images_path}")

        # Create temporary structure for DatasetOrganizer
        # DatasetOrganizer expects a directory with images/ and labels/ subdirectories
        temp_raw_dir = category_root / "_temp_raw"
        temp_images = temp_raw_dir / "images"
        temp_labels = temp_raw_dir / "labels"

        import shutil
        try:
            # Create temp structure
            ensure_dir(temp_raw_dir)

            # Copy input data to temp structure
            if temp_images.exists():
                shutil.rmtree(temp_images)
            if temp_labels.exists():
                shutil.rmtree(temp_labels)

            shutil.copytree(raw_images_path, temp_images)
            shutil.copytree(raw_labels_path, temp_labels)

            # Run dataset organization
            organizer = DatasetOrganizer(str(category_data_root))

            train_count, val_count = organizer.split_dataset(
                str(temp_raw_dir),
                split_ratio=category_config['validation']['split_ratio'],
                shuffle=category_config['validation']['shuffle'],
                seed=category_config['validation']['random_seed']
            )
            logger.info(f"[{category_name}] Dataset prepared: {train_count} train, {val_count} val")

            # Move the generated dataset_config.yaml to the category root
            # DatasetOrganizer creates it at category_data_root/dataset_config.yaml
            generated_config = category_data_root / "dataset_config.yaml"
            if generated_config.exists():
                shutil.copy2(generated_config, category_dataset_config)
                logger.info(f"[{category_name}] Dataset config copied to {category_dataset_config}")
            else:
                logger.error(f"[{category_name}] Dataset config not found at {generated_config}")
                return False

        finally:
            # Cleanup temp directory
            if temp_raw_dir.exists():
                try:
                    shutil.rmtree(temp_raw_dir)
                except Exception as e:
                    logger.warning(f"[{category_name}] Failed to cleanup temp directory: {e}")

        # Step 2: Train model
        logger.info(f"[{category_name}] Step 2: Training model...")
        trainer = YOLOTrainer(category_config)
        trainer.load_model()
        
        # Override project path for this category
        ensure_dir(category_model_dir)
        results = trainer.train(str(category_dataset_config))
        logger.info(f"[{category_name}] Training completed")
        
        # Step 3: Auto-annotate unlabeled data if exists
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
            logger.info(f"[{category_name}] No unlabeled data found at {unlabeled_dir}, skipping auto-annotation")
        
        logger.info(f"[{category_name}] [OK] Category processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"[{category_name}] [FAIL] Error processing category: {str(e)}", exc_info=True)
        return False


def main():
    """Main function to process categories in default or custom path mode"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Batch training for multiple categories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default mode: Process all categories in data/raw/
  python scripts/train_by_category.py

  # Custom path mode: Process all categories in a custom root path
  python scripts/train_by_category.py --data-root /path/to/data

Default mode structure (data/raw/):
  data/raw/
  ├── category1/
  │   ├── images/          # Input: labeled images
  │   └── labels/          # Input: labels
  ├── category1_unlabeled/
  │   └── images/          # Input: unlabeled images (optional)
  └── category2/
      ├── images/
      └── labels/

Custom path mode structure:
  /path/to/data/
  ├── category1/
  │   ├── pre_images/      # Input: labeled images
  │   ├── pre_labels/      # Input: labels
  │   ├── images/          # Input: unlabeled images (optional)
  │   ├── category/        # Output: train/val split (auto-created)
  │   ├── models/          # Output: trained models (auto-created)
  │   └── labels/          # Output: auto-generated labels (auto-created)
  └── category2/
      ├── pre_images/
      ├── pre_labels/
      └── ...
        """
    )
    parser.add_argument(
        '--data-root',
        type=str,
        help='Custom data root path containing category subdirectories with pre_images/ and pre_labels/'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(__name__, "logs/train_by_category.log")

    # Load base configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return

    base_config = load_config(args.config)
    logger.info(f"Loaded base configuration from {args.config}")

    # Determine mode and scan for categories
    if args.data_root:
        # Custom path mode
        logger.info("="*60)
        logger.info("CUSTOM PATH MODE")
        logger.info("="*60)

        data_root = Path(args.data_root).resolve()
        if not data_root.exists():
            logger.error(f"Data root directory not found: {data_root}")
            return

        logger.info(f"Scanning for categories in: {data_root}")

        # Find all category directories (must contain pre_images/ and pre_labels/ subdirectories)
        categories = []
        for item in data_root.iterdir():
            if item.is_dir():
                pre_images_dir = item / "pre_images"
                pre_labels_dir = item / "pre_labels"
                if pre_images_dir.exists() and pre_labels_dir.exists():
                    categories.append((item.name, item))

        if not categories:
            logger.warning(f"No valid categories found in {data_root}")
            logger.info("Each category should have 'pre_images/' and 'pre_labels/' subdirectories")
            return

        logger.info(f"Found {len(categories)} categories to process: {[c[0] for c in categories]}")

        # Process each category
        results = {}
        for category_name, category_path in categories:
            success = process_category(
                category_name,
                category_path,
                base_config,
                logger,
                use_pre_prefix=True
            )
            results[category_name] = success

    else:
        # Default mode: scan data/raw/
        logger.info("="*60)
        logger.info("DEFAULT MODE: Batch training by category")
        logger.info("="*60)

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
            success = process_category(
                category_name,
                category_path,
                base_config,
                logger,
                use_pre_prefix=False
            )
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
        status = "[OK]" if success else "[FAIL]"
        logger.info(f"  {status} {category}")

    logger.info("Batch training completed")


if __name__ == "__main__":
    main()