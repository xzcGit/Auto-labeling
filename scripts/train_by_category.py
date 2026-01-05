"""Batch processing script for multiple categories.

Supports two modes:
1) Default mode: scan and process all categories in data/raw/
2) Custom path mode: scan and process all categories in a custom root path

Enhancements:
- Can annotate directly with pretrained weights (no training needed)
- Can reuse existing trained weights to avoid retraining
- Can save/load a lightweight per-category model registry for reuse
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main function to process categories in default or custom path mode"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Batch processing for multiple categories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default mode: Process all categories in data/raw/
  python scripts/train_by_category.py

  # Custom path mode: Process all categories in a custom root path
  python scripts/train_by_category.py --data-root /path/to/data

  # Annotate only (reuse existing weights / registry / pretrained)
  python scripts/train_by_category.py --action annotate --pretrained-root /path/to/pretrained_models

  # Force retraining even if a model already exists
  python scripts/train_by_category.py --force-train

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
    parser.add_argument(
        '--action',
        type=str,
        choices=['train', 'annotate', 'train_and_annotate'],
        default='train_and_annotate',
        help='What to do for each category (default: train_and_annotate)'
    )
    parser.add_argument(
        '--force-train',
        action='store_true',
        help='Force retraining even if existing weights are found'
    )
    parser.add_argument(
        '--shared-model-root',
        type=str,
        default=None,
        help='Store and reuse models under <root>/<category>/ (improves reuse across runs)'
    )
    parser.add_argument(
        '--registry',
        type=str,
        default='models/model_registry.yaml',
        help='Model registry path used to resolve/update per-category weights'
    )
    parser.add_argument(
        '--model-map',
        type=str,
        default=None,
        help='YAML/JSON mapping file: {category: /path/to/weights.pt}'
    )
    parser.add_argument(
        '--pretrained-root',
        type=str,
        default=None,
        help='Directory containing pretrained weights per category'
    )
    parser.add_argument(
        '--pretrained-model',
        type=str,
        default=None,
        help='Single pretrained weights (.pt) used for all categories'
    )
    parser.add_argument(
        '--prefer-pretrained',
        action='store_true',
        help='Prefer pretrained sources over trained weights when both exist'
    )
    parser.add_argument(
        '--train-init',
        type=str,
        choices=['base', 'reuse'],
        default='base',
        help="Training init strategy when training is needed: 'base' (default) or 'reuse' (warm-start from resolved weights)"
    )
    parser.add_argument(
        '--output-layout',
        type=str,
        choices=['triage', 'yolo'],
        default='triage',
        help="Annotation output layout: 'triage' (default, confidence folders) or 'yolo' (labels/*.txt)"
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help="When --output-layout yolo, do not skip images that already have labels/*.txt"
    )

    args = parser.parse_args()

    from src.utils import load_config, setup_logger
    from src.category_pipeline import load_model_map
    from src.category_runner import process_category
    from src.model_registry import load_model_registry

    # Setup logger
    logger = setup_logger(__name__, "logs/train_by_category.log")

    # Load base configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return

    base_config = load_config(args.config)
    logger.info(f"Loaded base configuration from {args.config}")

    model_map = load_model_map(args.model_map) if args.model_map else {}
    registry = load_model_registry(args.registry) if args.registry else {}

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
                category_name=category_name,
                category_root=category_path,
                base_config=base_config,
                logger=logger,
                use_pre_prefix=True,
                action=args.action,
                force_train=args.force_train,
                train_init=args.train_init,
                shared_model_root=args.shared_model_root,
                registry_path=args.registry,
                registry=registry,
                model_map_path=args.model_map,
                model_map=model_map,
                pretrained_root=args.pretrained_root,
                pretrained_model=args.pretrained_model,
                prefer_pretrained=args.prefer_pretrained,
                output_layout=args.output_layout,
                skip_existing=not args.no_skip_existing,
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
                category_name=category_name,
                category_root=category_path,
                base_config=base_config,
                logger=logger,
                use_pre_prefix=False,
                action=args.action,
                force_train=args.force_train,
                train_init=args.train_init,
                shared_model_root=args.shared_model_root,
                registry_path=args.registry,
                registry=registry,
                model_map_path=args.model_map,
                model_map=model_map,
                pretrained_root=args.pretrained_root,
                pretrained_model=args.pretrained_model,
                prefer_pretrained=args.prefer_pretrained,
                output_layout=args.output_layout,
                skip_existing=not args.no_skip_existing,
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
