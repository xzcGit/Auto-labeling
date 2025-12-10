"""Prepare and organize dataset for training"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processor import DatasetOrganizer
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to raw data directory (with images/ and labels/ subdirs)')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for organized dataset')
    parser.add_argument('--split-ratio', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle dataset before splitting')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)
    print(f"Raw data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Split ratio: {args.split_ratio}")
    print(f"Shuffle: {not args.no_shuffle}")
    print("=" * 60)
    
    # Initialize organizer
    organizer = DatasetOrganizer(args.output_dir)
    
    # Validate and split dataset
    try:
        train_count, val_count = organizer.split_dataset(
            raw_dir=args.data_dir,
            split_ratio=args.split_ratio,
            shuffle=not args.no_shuffle,
            seed=args.seed
        )
        
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"\nNext step: Run training with:")
        print(f"  python scripts/train_model.py --config config/config.yaml")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())