"""Auto-label images using trained YOLO model"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_annotator import AutoAnnotator
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description='Auto-label images using trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to directory containing unlabeled images')
    parser.add_argument('--output', type=str, default='output/predictions',
                       help='Output directory for generated labels')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--conf-threshold', type=float, default=None,
                       help='Confidence threshold (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override confidence threshold if provided
    if args.conf_threshold:
        config['auto_annotation']['confidence_threshold'] = args.conf_threshold
    
    print("=" * 60)
    print("Auto-Annotation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")
    print(f"Confidence threshold: {config['auto_annotation']['confidence_threshold']}")
    print(f"Review threshold: {config['auto_annotation']['review_threshold']}")
    print("=" * 60)
    
    # Initialize annotator
    annotator = AutoAnnotator(args.model, config)
    
    try:
        start_time = datetime.now()
        print(f"\nAuto-annotation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Annotate images
        stats = annotator.annotate_images(args.images, args.output)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'=' * 60}")
        print(f"✓ Auto-annotation completed!")
        print(f"  Duration: {duration}")
        print(f"  Total images: {stats['total']}")
        print(f"  High confidence (>0.7): {stats['high_conf']} ({stats['high_conf']/stats['total']*100:.1f}%)")
        print(f"  Medium confidence (0.5-0.7): {stats['medium_conf']} ({stats['medium_conf']/stats['total']*100:.1f}%)")
        print(f"  Low confidence (<0.5): {stats['low_conf']} ({stats['low_conf']/stats['total']*100:.1f}%)")
        print(f"\nLabels saved to:")
        print(f"  High conf: {args.output}/labels/high_conf/")
        print(f"  Medium conf: {args.output}/labels/medium_conf/")
        print(f"  Low conf (review needed): {args.output}/labels/low_conf/")
        print(f"\nStatistics saved to: {args.output}/statistics.json")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during auto-annotation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())