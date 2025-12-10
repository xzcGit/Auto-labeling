"""Complete pipeline for training and auto-annotation"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from src.data_processor import DatasetOrganizer
from src.trainer import YOLOTrainer
from src.auto_annotator import AutoAnnotator
from src.utils import load_config, setup_logger


def main():
    parser = argparse.ArgumentParser(description='Run complete auto-annotation pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['prepare', 'train', 'annotate', 'full'],
                       help='Pipeline mode to run')
    parser.add_argument('--raw-data', type=str, default='data/raw',
                       help='Path to raw data (for prepare mode)')
    parser.add_argument('--unlabeled', type=str, default='data/unlabeled/images',
                       help='Path to unlabeled images (for annotate mode)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (for annotate mode)')
    
    args = parser.parse_args()
    
    logger = setup_logger('pipeline')
    config = load_config(args.config)
    
    print("=" * 70)
    print("AUTO-ANNOTATION PIPELINE")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Stage 1: Prepare data
        if args.mode in ['prepare', 'full']:
            print("\n[Stage 1/3] Preparing dataset...")
            print("-" * 70)
            organizer = DatasetOrganizer(config['paths']['data_root'])
            train_count, val_count = organizer.split_dataset(
                raw_dir=args.raw_data,
                split_ratio=config['validation']['split_ratio'],
                shuffle=config['validation']['shuffle'],
                seed=config['validation']['random_seed']
            )
            print(f"✓ Dataset prepared: {train_count} train, {val_count} val samples")
        
        # Stage 2: Train model
        if args.mode in ['train', 'full']:
            print("\n[Stage 2/3] Training model...")
            print("-" * 70)
            trainer = YOLOTrainer(config)
            dataset_config = 'config/dataset_config.yaml'
            results = trainer.train(dataset_config)
            print(f"✓ Training completed")
            
            # Get best model path
            model_path = Path(config['paths']['model_root']) / 'trained' / 'train' / 'weights' / 'best.pt'
        else:
            model_path = args.model
        
        # Stage 3: Auto-annotate
        if args.mode in ['annotate', 'full']:
            print("\n[Stage 3/3] Auto-annotating images...")
            print("-" * 70)
            if not model_path:
                raise ValueError("Model path required for annotation. Use --model or run full pipeline.")
            
            annotator = AutoAnnotator(str(model_path), config)
            stats = annotator.annotate_images(
                args.unlabeled,
                config['paths']['output_root'] + '/predictions'
            )
            print(f"✓ Auto-annotation completed: {stats['total']} images processed")
        
        # Success summary
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.mode == 'full':
            print("\nOutput locations:")
            print(f"  Prepared data: {config['paths']['data_root']}/")
            print(f"  Trained model: {config['paths']['model_root']}/trained/")
            print(f"  Auto-labels: {config['paths']['output_root']}/predictions/labels/")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())