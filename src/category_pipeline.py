"""Category-level pipeline utilities.

Used by scripts to prepare data, train models, and/or auto-annotate per category.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .auto_annotator import AutoAnnotator
from .data_processor import DatasetOrganizer
from .trainer import YOLOTrainer
from .utils import ensure_dir
from .model_registry import resolve_registry_weight


@dataclass(frozen=True)
class CategoryIO:
    category_name: str
    raw_images_dir: Path
    raw_labels_dir: Path
    unlabeled_images_dir: Optional[Path]
    data_root: Path
    model_root: Path
    output_root: Path
    dataset_config_path: Path


def load_model_map(model_map_path: Optional[str]) -> Dict[str, str]:
    if not model_map_path:
        return {}

    path = Path(model_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Model map not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid model map format (expected mapping): {path}")

    model_map: Dict[str, str] = {}
    for category, weights in data.items():
        if category is None or weights is None:
            continue
        model_map[str(category)] = str(weights)
    return model_map


def resolve_model_from_map(
    model_map: Dict[str, str],
    category_name: str,
    model_map_path: Optional[str] = None
) -> Optional[Path]:
    value = model_map.get(category_name)
    if not value:
        return None

    p = Path(value)
    if p.is_absolute():
        return p if p.exists() else None

    if model_map_path:
        candidate = (Path(model_map_path).parent / p).resolve()
        return candidate if candidate.exists() else None

    candidate = p.resolve()
    return candidate if candidate.exists() else None


def resolve_model_from_root(pretrained_root: Optional[str], category_name: str) -> Optional[Path]:
    if not pretrained_root:
        return None

    root = Path(pretrained_root)
    candidates = [
        root / f"{category_name}.pt",
        root / category_name / "best.pt",
        root / category_name / "train" / "weights" / "best.pt",
        root / category_name / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_trained_best_weights(model_root: Path) -> Optional[Path]:
    best = model_root / "train" / "weights" / "best.pt"
    return best if best.exists() else None


def resolve_weights(
    category_name: str,
    model_root: Path,
    *,
    registry: Optional[Dict[str, str]] = None,
    registry_path: Optional[str] = None,
    model_map: Optional[Dict[str, str]] = None,
    model_map_path: Optional[str] = None,
    pretrained_root: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    prefer_pretrained: bool = False
) -> Tuple[Optional[Path], str]:
    """Resolve weights for category with a deterministic priority order."""
    trained = find_trained_best_weights(model_root)
    registry_weight = resolve_registry_weight(registry or {}, category_name, registry_path=registry_path)
    map_weight = resolve_model_from_map(model_map or {}, category_name, model_map_path=model_map_path)
    root_weight = resolve_model_from_root(pretrained_root, category_name)
    single_weight = Path(pretrained_model) if pretrained_model else None
    if single_weight and not single_weight.exists():
        single_weight = None

    if prefer_pretrained:
        for p, source in [
            (map_weight, "model_map"),
            (root_weight, "pretrained_root"),
            (single_weight, "pretrained_model"),
            (registry_weight, "registry"),
            (trained, "trained"),
        ]:
            if p:
                return p, source
        return None, "missing"

    for p, source in [
        (trained, "trained"),
        (registry_weight, "registry"),
        (map_weight, "model_map"),
        (root_weight, "pretrained_root"),
        (single_weight, "pretrained_model"),
    ]:
        if p:
            return p, source
    return None, "missing"


def build_category_io(
    category_name: str,
    category_root: Path,
    *,
    use_pre_prefix: bool,
    shared_model_root: Optional[str] = None
) -> CategoryIO:
    if use_pre_prefix:
        raw_images_dir = category_root / "pre_images"
        raw_labels_dir = category_root / "pre_labels"
        unlabeled_images_dir = category_root / "images"
    else:
        raw_images_dir = category_root / "images"
        raw_labels_dir = category_root / "labels"
        unlabeled_images_dir = category_root.parent / f"{category_name}_unlabeled" / "images"

    data_root = category_root / "category"
    if shared_model_root:
        model_root = Path(shared_model_root).expanduser().resolve() / category_name
    else:
        model_root = category_root / "models"
    output_root = category_root / "labels"
    dataset_config_path = category_root / "dataset_config.yaml"

    return CategoryIO(
        category_name=category_name,
        raw_images_dir=raw_images_dir,
        raw_labels_dir=raw_labels_dir,
        unlabeled_images_dir=unlabeled_images_dir,
        data_root=data_root,
        model_root=model_root,
        output_root=output_root,
        dataset_config_path=dataset_config_path,
    )


def prepare_dataset(io: CategoryIO, config: Dict[str, Any], logger) -> None:
    ensure_dir(str(io.data_root))
    organizer = DatasetOrganizer(str(io.data_root))
    train_count, val_count = organizer.split_dataset_from_dirs(
        images_dir=str(io.raw_images_dir),
        labels_dir=str(io.raw_labels_dir),
        split_ratio=config["validation"]["split_ratio"],
        shuffle=config["validation"]["shuffle"],
        seed=config["validation"]["random_seed"],
    )
    logger.info(f"[{io.category_name}] Dataset prepared: {train_count} train, {val_count} val")

    generated_config = io.data_root / "dataset_config.yaml"
    if generated_config.exists():
        io.dataset_config_path.write_text(generated_config.read_text(encoding="utf-8"), encoding="utf-8")


def train_model(
    io: CategoryIO,
    config: Dict[str, Any],
    logger,
    *,
    init_weights: Optional[Path] = None,
) -> Path:
    ensure_dir(str(io.model_root))
    trainer = YOLOTrainer(config)
    trainer.load_model(str(init_weights) if init_weights else None)
    trainer.train(str(io.dataset_config_path))
    best = io.model_root / "train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Best weights not found after training: {best}")
    logger.info(f"[{io.category_name}] Best weights: {best}")
    return best


def auto_annotate(
    io: CategoryIO,
    config: Dict[str, Any],
    weights_path: Path,
    logger,
    *,
    output_layout: str = "triage",
    skip_existing: bool = True,
):
    if not io.unlabeled_images_dir or not io.unlabeled_images_dir.exists():
        logger.info(f"[{io.category_name}] No unlabeled data found, skipping auto-annotation")
        return None

    annotator = AutoAnnotator(str(weights_path), config)
    if output_layout == "yolo":
        ensure_dir(str(io.output_root))
        stats = annotator.annotate_images_yolo(
            str(io.unlabeled_images_dir),
            str(io.output_root),
            skip_existing=skip_existing,
            write_empty=True,
            report_path=str(io.output_root / "_auto_label_report.json"),
        )
    elif output_layout == "triage":
        ensure_dir(str(io.output_root))
        stats = annotator.annotate_images(str(io.unlabeled_images_dir), str(io.output_root))
    else:
        raise ValueError(f"Unsupported output_layout: {output_layout}")
    logger.info(f"[{io.category_name}] Auto-annotation completed: {stats}")
    return stats
