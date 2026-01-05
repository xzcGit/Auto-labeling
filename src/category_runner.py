"""High-level runner for processing a single category.

This module centralizes the orchestration logic so multiple CLI scripts can
share consistent behavior (train/annotate, model reuse, output layout).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from .category_pipeline import (
    auto_annotate,
    build_category_io,
    prepare_dataset,
    resolve_weights,
    train_model,
)
from .model_registry import update_registry_for_category


def process_category(
    *,
    category_name: str,
    category_root: Path,
    base_config: Dict[str, Any],
    logger,
    use_pre_prefix: bool,
    action: str = "train_and_annotate",
    force_train: bool = False,
    train_init: str = "base",
    shared_model_root: Optional[str] = None,
    registry_path: Optional[str] = None,
    registry: Optional[Dict[str, str]] = None,
    model_map_path: Optional[str] = None,
    model_map: Optional[Dict[str, str]] = None,
    pretrained_root: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    prefer_pretrained: bool = False,
    output_layout: str = "triage",
    skip_existing: bool = True,
) -> bool:
    """Process a single category: optionally train, and optionally auto-annotate.

    Notes:
    - When action is annotate-only, this runner does NOT require labeled data to exist.
    - When action includes training, it requires raw labeled dirs (images+labels) to exist.
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Category root: {category_root}")

        io = build_category_io(
            category_name,
            category_root,
            use_pre_prefix=use_pre_prefix,
            shared_model_root=shared_model_root,
        )

        should_train = action in ("train", "train_and_annotate")
        should_annotate = action in ("annotate", "train_and_annotate")

        if should_train:
            if not io.raw_images_dir.exists():
                logger.error(f"[{category_name}] Input images directory not found: {io.raw_images_dir}")
                return False
            if not io.raw_labels_dir.exists():
                logger.error(f"[{category_name}] Input labels directory not found: {io.raw_labels_dir}")
                return False

        category_config = copy.deepcopy(base_config)
        category_config["paths"]["data_root"] = str(io.data_root)
        category_config["paths"]["model_root"] = str(io.model_root)
        category_config["paths"]["output_root"] = str(io.output_root)

        weights_path: Optional[Path] = None

        if should_train:
            existing, source = resolve_weights(
                category_name,
                io.model_root,
                registry=registry,
                registry_path=registry_path,
                model_map=model_map,
                model_map_path=model_map_path,
                pretrained_root=pretrained_root,
                pretrained_model=pretrained_model,
                prefer_pretrained=prefer_pretrained,
            )

            if existing and not force_train and source in ("trained", "registry"):
                logger.info(f"[{category_name}] Reusing existing weights (skip training): {existing}")
                weights_path = existing
            else:
                logger.info(f"[{category_name}] Step 1: Preparing dataset...")
                prepare_dataset(io, category_config, logger)

                init_weights = existing if (train_init == "reuse" and existing) else None
                if init_weights:
                    logger.info(f"[{category_name}] Training init weights: {init_weights} (source={source})")

                logger.info(f"[{category_name}] Step 2: Training model...")
                weights_path = train_model(io, category_config, logger, init_weights=init_weights)

                if registry_path:
                    update_registry_for_category(registry_path, category_name, str(weights_path))

        if should_annotate:
            if not weights_path:
                resolved, source = resolve_weights(
                    category_name,
                    io.model_root,
                    registry=registry,
                    registry_path=registry_path,
                    model_map=model_map,
                    model_map_path=model_map_path,
                    pretrained_root=pretrained_root,
                    pretrained_model=pretrained_model,
                    prefer_pretrained=prefer_pretrained,
                )
                if not resolved:
                    # First-run ergonomics: if the user requested annotate-only but no weights
                    # are available, and labeled data exists, train once to unblock annotation.
                    # This is intentionally narrow to avoid surprising long training runs.
                    can_train_here = io.raw_images_dir.exists() and io.raw_labels_dir.exists()
                    if action == "annotate" and can_train_here:
                        logger.warning(
                            f"[{category_name}] No weights found for annotation (source={source}); "
                            f"found labeled data, auto-training a model to proceed"
                        )
                        logger.info(f"[{category_name}] Step 1: Preparing dataset...")
                        prepare_dataset(io, category_config, logger)
                        logger.info(f"[{category_name}] Step 2: Training model...")
                        weights_path = train_model(io, category_config, logger, init_weights=None)
                        if registry_path:
                            update_registry_for_category(registry_path, category_name, str(weights_path))
                    else:
                        logger.error(
                            f"[{category_name}] No usable weights found for annotation (source={source}). "
                            f"Provide weights via --shared-model-root/registry/model-map/pretrained-* "
                            f"or run with action=train_and_annotate"
                        )
                        return False
                else:
                    logger.info(f"[{category_name}] Using weights from {source}: {resolved}")
                    weights_path = resolved

            logger.info(f"[{category_name}] Step 3: Auto-annotating unlabeled data...")
            auto_annotate(
                io,
                category_config,
                weights_path,
                logger,
                output_layout=output_layout,
                skip_existing=skip_existing,
            )

        logger.info(f"[{category_name}] [OK] Category processing completed successfully")
        return True

    except Exception as e:
        logger.error(f"[{category_name}] [FAIL] Error processing category: {str(e)}", exc_info=True)
        return False
