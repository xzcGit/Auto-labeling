"""Train YOLO model on prepared dataset"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data', type=str, default='config/dataset_config.yaml',
                       help='Path to dataset config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu, overrides config)')
    
    args = parser.parse_args()

    from src.trainer import YOLOTrainer
    from src.utils import load_config

    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.device:
        config['training']['device'] = args.device
    
    print("=" * 60)
    print("YOLO Model Training")
    print("=" * 60)
    print(f"Model: {config['training']['model_type']}{config['training']['model_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Image size: {config['training']['img_size']}")
    print(f"Device: {config['training']['device']}")
    print(f"Dataset config: {args.data}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = YOLOTrainer(config)
    
    try:
        # Train model
        start_time = datetime.now()
        print(f"\nTraining started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = trainer.train(args.data)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'=' * 60}")
        print(f"✓ Training completed!")
        print(f"  Duration: {duration}")
        model_root = config["paths"]["model_root"]
        best_weights = f"{model_root}/train/weights/best.pt"
        print(f"  Model root: {model_root}")
        print(f"  Best weights: {best_weights}")
        print(f"\nNext step: Run auto-annotation with:")
        print(f"  python3 scripts/auto_label.py --model \"{best_weights}\" --images \"data/unlabeled/images\"")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
