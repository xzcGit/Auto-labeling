"""Batch processing script for metertools-style station datasets.

Goal:
- Stations can be added any time under a single root directory.
- For each station, scan categories and auto-annotate new images.
- Reuse already-trained category models across stations via a shared model root
  and/or registry.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process metertools-style station datasets (train and/or annotate per category)."
    )
    parser.add_argument(
        "--stations-root",
        type=str,
        required=True,
        help="Root directory that contains station subdirectories (e.g. /mnt/f/code/utils/19-metertools)",
    )
    parser.add_argument(
        "--station",
        type=str,
        action="append",
        default=None,
        help="Optional station name; can be repeated. When omitted, process all stations under --stations-root.",
    )
    parser.add_argument(
        "--category",
        type=str,
        action="append",
        default=None,
        help="Optional category name filter; can be repeated. When omitted, process all scanned categories.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["train", "annotate", "train_and_annotate"],
        default="annotate",
        help="What to do for each category (default: annotate)",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force retraining even if existing weights are found",
    )
    parser.add_argument(
        "--train-init",
        type=str,
        choices=["base", "reuse"],
        default="reuse",
        help="Training init strategy: 'base' or 'reuse' (default: reuse)",
    )
    parser.add_argument(
        "--shared-model-root",
        type=str,
        default="models/shared",
        help="Shared model root for cross-station reuse (default: models/shared)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="models/model_registry.yaml",
        help="Model registry path used to resolve/update per-category weights",
    )
    parser.add_argument(
        "--model-map",
        type=str,
        default=None,
        help="YAML/JSON mapping file: {category: /path/to/weights.pt}",
    )
    parser.add_argument(
        "--pretrained-root",
        type=str,
        default=None,
        help="Directory containing pretrained weights per category",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Single pretrained weights (.pt) used for all categories",
    )
    parser.add_argument(
        "--prefer-pretrained",
        action="store_true",
        help="Prefer pretrained sources over trained weights when both exist",
    )
    parser.add_argument(
        "--output-layout",
        type=str,
        choices=["yolo", "triage"],
        default="yolo",
        help="Annotation output layout for scanned station categories (default: yolo)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="When output layout is yolo, do not skip images that already have labels/*.txt",
    )

    args = parser.parse_args()

    from src.utils import load_config, setup_logger, ensure_dir
    from src.category_pipeline import load_model_map, resolve_weights
    from src.category_runner import process_category
    from src.model_registry import load_model_registry
    from src.auto_annotator import AutoAnnotator
    from src.station_scanner import iter_station_dirs, scan_station_categories

    logger = setup_logger(__name__, "logs/train_by_station.log")

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    base_config = load_config(args.config)
    logger.info(f"Loaded base configuration from {args.config}")

    model_map = load_model_map(args.model_map) if args.model_map else {}
    registry = load_model_registry(args.registry) if args.registry else {}

    stations_root = Path(args.stations_root).resolve()
    if not stations_root.exists():
        logger.error(f"Stations root directory not found: {stations_root}")
        return 1

    station_dirs = iter_station_dirs(stations_root)
    if args.station:
        wanted = set(args.station)
        station_dirs = [p for p in station_dirs if p.name in wanted]
        missing = wanted - {p.name for p in station_dirs}
        if missing:
            logger.warning(f"Stations not found under root: {sorted(missing)}")

    category_filter = set(args.category or [])

    shared_model_root = args.shared_model_root
    if args.shared_model_root == "models/shared":
        default_shared = Path(args.shared_model_root).expanduser().resolve()
        legacy_trained = Path("models/trained").resolve()
        if not default_shared.exists() and legacy_trained.exists():
            logger.warning(
                f"Shared model root not found at {default_shared}; "
                f"fall back to legacy trained root: {legacy_trained}"
            )
            shared_model_root = str(legacy_trained)

    ensure_dir("logs")
    logger.info("=" * 60)
    logger.info("STATION MODE: Batch by station")
    logger.info("=" * 60)
    logger.info(f"Stations root: {stations_root}")
    logger.info(f"Stations selected: {[p.name for p in station_dirs]}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Shared model root: {Path(shared_model_root).expanduser().resolve()}")
    logger.info(f"Output layout: {args.output_layout}")

    results = {}
    for station_dir in station_dirs:
        station_name = station_dir.name
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Station: {station_name}")
        logger.info(f"{'#' * 60}")

        categories = scan_station_categories(station_dir)
        if category_filter:
            categories = [c for c in categories if c.category_name in category_filter]

        if not categories:
            logger.warning(f"[{station_name}] No categories found")
            continue

        logger.info(f"[{station_name}] Found {len(categories)} categories: {[c.category_name for c in categories]}")

        for entry in categories:
            category_name = entry.category_name
            category_dir = entry.category_dir

            effective_action = args.action
            has_pre_labeled = (category_dir / "pre_images").exists() and (category_dir / "pre_labels").exists()
            if effective_action in ("train", "train_and_annotate") and not has_pre_labeled:
                logger.info(
                    f"[{station_name}/{category_name}] No pre_images/pre_labels; downgrade action to annotate"
                )
                effective_action = "annotate"

            if entry.layout in ("dir_images", "pre_labeled"):
                ok = process_category(
                    category_name=category_name,
                    category_root=category_dir,
                    base_config=base_config,
                    logger=logger,
                    use_pre_prefix=True,
                    action=effective_action,
                    force_train=args.force_train,
                    train_init=args.train_init,
                    shared_model_root=shared_model_root,
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
                results[(station_name, category_name)] = ok
                continue

            if entry.layout == "flat_images":
                model_root = Path(shared_model_root).expanduser().resolve() / category_name
                weights, source = resolve_weights(
                    category_name,
                    model_root,
                    registry=registry,
                    registry_path=args.registry,
                    model_map=model_map,
                    model_map_path=args.model_map,
                    pretrained_root=args.pretrained_root,
                    pretrained_model=args.pretrained_model,
                    prefer_pretrained=args.prefer_pretrained,
                )
                if not weights:
                    logger.error(f"[{station_name}/{category_name}] No usable weights found (source={source})")
                    results[(station_name, category_name)] = False
                    continue

                labels_dir = category_dir / "labels"
                annotator = AutoAnnotator(str(weights), base_config)
                logger.info(f"[{station_name}/{category_name}] Flat images -> labels dir: {labels_dir}")
                annotator.annotate_images_yolo(
                    str(category_dir),
                    str(labels_dir),
                    skip_existing=not args.no_skip_existing,
                    write_empty=True,
                    report_path=str(labels_dir / "_auto_label_report.json"),
                )
                results[(station_name, category_name)] = True
                continue

            logger.warning(f"[{station_name}/{category_name}] Unknown layout: {entry.layout}")
            results[(station_name, category_name)] = False

    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    logger.info(f"\n{'=' * 60}")
    logger.info("Station Batch Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total station/category tasks: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
