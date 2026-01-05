"""Auto-label images using trained YOLO model"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Auto-label images using trained model')
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str,
                            help='Path to model weights (.pt file)')
    model_group.add_argument('--category', type=str,
                            help='Category name to resolve model from registry')
    parser.add_argument('--registry', type=str, default='models/model_registry.yaml',
                       help='Model registry used when --category is provided')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to directory containing unlabeled images')
    parser.add_argument('--output', type=str, default='output/predictions',
                       help='Output directory for generated labels')
    parser.add_argument(
        '--output-layout',
        type=str,
        choices=['triage', 'yolo'],
        default='triage',
        help="Output layout: 'triage' (default, confidence folders) or 'yolo' (labels/*.txt)"
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help="When --output-layout yolo, do not skip images that already have labels/*.txt"
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--conf-threshold', type=float, default=None,
                       help='Confidence threshold (overrides config)')
    
    args = parser.parse_args()

    from src.auto_annotator import AutoAnnotator
    from src.utils import load_config
    from src.model_registry import load_model_registry, resolve_registry_weight

    # Load config
    config = load_config(args.config)
    
    # Override confidence threshold if provided
    if args.conf_threshold:
        config['auto_annotation']['confidence_threshold'] = args.conf_threshold
    
    print("=" * 60)
    print("Auto-Annotation")
    print("=" * 60)
    model_path = args.model
    if args.category:
        registry = load_model_registry(args.registry)
        resolved = resolve_registry_weight(registry, args.category, registry_path=args.registry)
        if not resolved:
            print(f"✗ No model found for category '{args.category}' in registry: {args.registry}")
            return 1
        model_path = str(resolved)

    print(f"Model: {model_path}")
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")
    print(f"Output layout: {args.output_layout}")
    print(f"Confidence threshold: {config['auto_annotation']['confidence_threshold']}")
    print(f"Review threshold: {config['auto_annotation']['review_threshold']}")
    print("=" * 60)
    
    # Initialize annotator
    annotator = AutoAnnotator(model_path, config)
    
    try:
        start_time = datetime.now()
        print(f"\nAuto-annotation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Annotate images
        if args.output_layout == "yolo":
            stats = annotator.annotate_images_yolo(
                args.images,
                args.output,
                skip_existing=not args.no_skip_existing,
                write_empty=True,
                report_path=str(Path(args.output) / "_auto_label_report.json"),
            )
        else:
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
        if args.output_layout == "yolo":
            print(f"\nLabels saved to: {args.output}/*.txt")
            print(f"Report saved to: {args.output}/_auto_label_report.json")
        else:
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
