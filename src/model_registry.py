"""Model registry helpers.

Keep a lightweight mapping from category name -> model weights path, so that a
previously trained model can be reused for future auto-annotation runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

from .utils import ensure_dir


def load_model_registry(registry_path: str) -> Dict[str, str]:
    path = Path(registry_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid registry format (expected mapping): {path}")

    # Normalize to str->str
    registry: Dict[str, str] = {}
    for category, weights in data.items():
        if category is None or weights is None:
            continue
        registry[str(category)] = str(weights)
    return registry


def save_model_registry(registry: Dict[str, str], registry_path: str) -> None:
    path = Path(registry_path)
    ensure_dir(str(path.parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, allow_unicode=True, sort_keys=True)


def resolve_registry_weight(
    registry: Dict[str, str],
    category_name: str,
    registry_path: Optional[str] = None
) -> Optional[Path]:
    value = registry.get(category_name)
    if not value:
        return None

    p = Path(value)
    if p.is_absolute():
        return p if p.exists() else None

    if registry_path:
        candidate = (Path(registry_path).parent / p).resolve()
        return candidate if candidate.exists() else None

    candidate = p.resolve()
    return candidate if candidate.exists() else None


def update_registry_for_category(
    registry_path: str,
    category_name: str,
    weights_path: str
) -> None:
    registry = load_model_registry(registry_path)
    registry[category_name] = str(weights_path)
    save_model_registry(registry, registry_path)
