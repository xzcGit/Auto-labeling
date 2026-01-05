"""Station dataset scanning helpers.

For the "metertools" style dataset, the hierarchy is usually:
  stations_root/
    <station_name>/
      det/ (optional)
        <category_name>/
          pre_images/ pre_labels/ (optional, labeled)
          images/ (optional, unlabeled)
          labels/ (optional, existing labels)
      <category_name>/  (some stations are flat folders with only images)
        *.jpg/*.jpeg/...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .utils import get_image_files


_DEFAULT_IGNORE_DIRS = {
    "cls",
    "det",
    "models",
    "dataset",
    "config",
    "category",
    "output",
    "logs",
    # Station-root "category-style" layouts often contain these reserved dirs.
    # They are not categories by themselves.
    "images",
    "labels",
    "pre_images",
    "pre_labels",
    "__pycache__",
}


@dataclass(frozen=True)
class StationCategory:
    station_name: str
    category_name: str
    category_dir: Path
    layout: str  # "dir_images" | "pre_labeled" | "flat_images"


def iter_station_dirs(stations_root: Path) -> List[Path]:
    stations_root = stations_root.resolve()
    if not stations_root.exists():
        raise FileNotFoundError(f"Stations root not found: {stations_root}")
    return sorted([p for p in stations_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def scan_station_categories(
    station_dir: Path,
    *,
    ignore_dirs: Optional[Iterable[str]] = None,
) -> List[StationCategory]:
    station_dir = station_dir.resolve()
    ignore = set(ignore_dirs or _DEFAULT_IGNORE_DIRS)

    scan_roots = []
    det_root = station_dir / "det"
    if det_root.exists() and det_root.is_dir():
        scan_roots.append(det_root)
    scan_roots.append(station_dir)

    found: List[StationCategory] = []
    seen = set()

    # Support a common "category-style station" variant where the station dir
    # itself contains pre_images/pre_labels and/or images (no nested categories).
    # Example:
    #   <stations_root>/<station>/{pre_images,pre_labels,images,...}
    station_has_root_layout = (
        (station_dir / "images").exists()
        or ((station_dir / "pre_images").exists() and (station_dir / "pre_labels").exists())
        or bool(get_image_files(str(station_dir)))
    )
    det_has_categories = det_root.exists() and any(
        p.is_dir() and not p.name.startswith(".") for p in det_root.iterdir()
    )
    station_has_nested_categories = any(
        p.is_dir()
        and not p.name.startswith(".")
        and p.name not in ignore
        and p.name != "det"
        for p in station_dir.iterdir()
    )

    if station_has_root_layout and not det_has_categories and not station_has_nested_categories:
        if (station_dir / "images").exists():
            layout = "dir_images"
        elif (station_dir / "pre_images").exists() and (station_dir / "pre_labels").exists():
            layout = "pre_labeled"
        else:
            layout = "flat_images"

        found.append(
            StationCategory(
                station_name=station_dir.name,
                category_name=station_dir.name,
                category_dir=station_dir,
                layout=layout,
            )
        )

    for root in scan_roots:
        for item in sorted(root.iterdir()):
            if not item.is_dir():
                continue
            if item.name.startswith("."):
                continue
            if root == station_dir and item.name in ignore:
                continue

            key = (item.name, str(item.resolve()))
            if key in seen:
                continue
            seen.add(key)

            if (item / "images").exists():
                found.append(
                    StationCategory(
                        station_name=station_dir.name,
                        category_name=item.name,
                        category_dir=item,
                        layout="dir_images",
                    )
                )
                continue

            if (item / "pre_images").exists() and (item / "pre_labels").exists():
                found.append(
                    StationCategory(
                        station_name=station_dir.name,
                        category_name=item.name,
                        category_dir=item,
                        layout="pre_labeled",
                    )
                )
                continue

            if get_image_files(str(item)):
                found.append(
                    StationCategory(
                        station_name=station_dir.name,
                        category_name=item.name,
                        category_dir=item,
                        layout="flat_images",
                    )
                )
                continue

    return found
