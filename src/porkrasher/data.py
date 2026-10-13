"""Dataset utilities for the pork rasher packaging-defect project.

The dataset is the Roboflow export *Pork Rasher Error (Packaging)*, version 4, in
YOLO format. Each split (``train``, ``valid``, ``test``) contains an ``images``
folder and a ``labels`` folder. Every image has one ``.txt`` label file with one
line per annotated defect::

    class_id x_center y_center width height          (bounding box, 4 values)
    class_id x1 y1 x2 y2 ... xn yn                   (polygon, even count >= 6)

All coordinates are normalised to [0, 1] by image width and height. An empty
label file means the image contains no defect ("background" image).

In the provided data every label is a plain bounding box (verified by the
Ultralytics loader, which found 0 polygons in 2,923 training labels), so the
project uses object *detection*, not segmentation. The parser still accepts
polygons so the helpers remain correct if the labels are ever upgraded.

Invalid input handling
----------------------
* Malformed label lines raise :class:`ValueError` by default (with the file name
  and line number). Pass ``skip_invalid=True`` to log a warning and skip them.
* Missing files or folders raise :class:`FileNotFoundError`.
* A zip that does not contain the expected split folders raises
  :class:`ValueError`.

Command line
------------
``python -m porkrasher.data --root data/dataset`` prints class counts per split,
source-frame overlap between splits, and per-class box-size statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

#: Class names in the order of their ids in the dataset's ``data.yaml``.
CLASS_NAMES: List[str] = [
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
]

#: Split folder names used by the dataset.
SPLITS: Tuple[str, ...] = ("train", "valid", "test")

#: File suffixes treated as images.
IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Tolerance for coordinates marginally outside [0, 1] due to float rounding.
_COORD_TOLERANCE = 1e-6

Point = Tuple[float, float]
Annotation = Tuple[int, List[Point]]
PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Label parsing
# --------------------------------------------------------------------------- #
def parse_label_line(
    line: str, n_classes: int = len(CLASS_NAMES)
) -> Optional[Annotation]:
    """Parse one line of a YOLO label file.

    Args:
        line: A single line such as ``"3 0.5 0.5 0.2 0.1"``.
        n_classes: Number of valid classes; ids must lie in ``[0, n_classes)``.

    Returns:
        ``None`` for a blank line. Otherwise ``(class_id, points)`` where
        ``points`` are normalised ``(x, y)`` vertices. A bounding box is
        returned as its four corners (top-left, top-right, bottom-right,
        bottom-left) so boxes and polygons can be handled identically.

    Raises:
        ValueError: If the class id is not an integer in range, any coordinate
            is non-numeric or outside [0, 1], or the number of coordinates is
            neither 4 (box) nor an even number >= 6 (polygon).
    """
    parts = line.split()
    if not parts:
        return None

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"non-numeric value in label line: {line!r}") from exc

    if not 0 <= class_id < n_classes:
        raise ValueError(
            f"class id {class_id} out of range [0, {n_classes}) in line: {line!r}"
        )
    if any(not -_COORD_TOLERANCE <= c <= 1 + _COORD_TOLERANCE for c in coords):
        raise ValueError(f"coordinate outside [0, 1] in label line: {line!r}")

    if len(coords) == 4:
        xc, yc, width, height = coords
        x1, y1 = xc - width / 2, yc - height / 2
        x2, y2 = xc + width / 2, yc + height / 2
        return class_id, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if len(coords) >= 6 and len(coords) % 2 == 0:
        return class_id, list(zip(coords[0::2], coords[1::2]))
    raise ValueError(
        f"expected 4 values (box) or an even number >= 6 (polygon), "
        f"got {len(coords)} in line: {line!r}"
    )


def load_yolo_label(
    label_path: PathLike,
    n_classes: int = len(CLASS_NAMES),
    skip_invalid: bool = False,
) -> List[Annotation]:
    """Read all annotations from a YOLO label file.

    Args:
        label_path: Path to a ``.txt`` label file.
        n_classes: Number of valid classes.
        skip_invalid: If True, log a warning and skip malformed lines instead
            of raising.

    Returns:
        A list of ``(class_id, points)`` tuples (see :func:`parse_label_line`).
        An empty list means a background image.

    Raises:
        FileNotFoundError: If ``label_path`` does not exist.
        ValueError: On a malformed line when ``skip_invalid`` is False. The
            message includes the file name and 1-based line number.
    """
    label_path = Path(label_path)
    annotations: List[Annotation] = []
    with label_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                parsed = parse_label_line(line, n_classes)
            except ValueError as exc:
                if skip_invalid:
                    logger.warning("%s:%d skipped: %s", label_path, line_number, exc)
                    continue
                raise ValueError(f"{label_path}:{line_number}: {exc}") from exc
            if parsed is not None:
                annotations.append(parsed)
    return annotations


def _extent(points: Sequence[Point]) -> Tuple[float, float]:
    """Width and height of the axis-aligned bounding extent of ``points``."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(xs) - min(xs), max(ys) - min(ys)


# --------------------------------------------------------------------------- #
# Dataset statistics
# --------------------------------------------------------------------------- #
@dataclass
class SplitSummary:
    """Class statistics for one dataset split.

    Attributes:
        n_label_files: Number of ``.txt`` label files (one per image).
        n_background: Label files with no annotations (defect-free images).
        counts: Instances per class name; classes never seen are absent
            (a :class:`collections.Counter` returns 0 for them).
    """

    n_label_files: int = 0
    n_background: int = 0
    counts: Counter = field(default_factory=Counter)

    @property
    def n_instances(self) -> int:
        """Total annotated instances in the split."""
        return sum(self.counts.values())


def summarise_split(
    label_dir: PathLike,
    class_names: Sequence[str] = CLASS_NAMES,
    skip_invalid: bool = False,
) -> SplitSummary:
    """Count label files, background images and instances per class.

    Args:
        label_dir: Folder of ``.txt`` label files for one split.
        class_names: Class names, indexed by class id.
        skip_invalid: Skip malformed lines instead of raising (see
            :func:`load_yolo_label`).

    Returns:
        A :class:`SplitSummary`.

    Raises:
        FileNotFoundError: If ``label_dir`` does not exist.
    """
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label folder not found: {label_dir}")

    summary = SplitSummary()
    for label_file in sorted(label_dir.glob("*.txt")):
        summary.n_label_files += 1
        annotations = load_yolo_label(label_file, len(class_names), skip_invalid)
        if not annotations:
            summary.n_background += 1
        for class_id, _ in annotations:
            summary.counts[class_names[class_id]] += 1
    return summary


def class_counts_by_split(
    root: PathLike,
    class_names: Sequence[str] = CLASS_NAMES,
    splits: Sequence[str] = SPLITS,
    skip_invalid: bool = False,
) -> Dict[str, SplitSummary]:
    """Run :func:`summarise_split` on every split of a dataset.

    Args:
        root: Dataset root containing one folder per split.
        class_names: Class names, indexed by class id.
        splits: Split folder names.
        skip_invalid: Skip malformed label lines instead of raising.

    Returns:
        Mapping from split name to its :class:`SplitSummary`.
    """
    root = Path(root)
    return {
        split: summarise_split(root / split / "labels", class_names, skip_invalid)
        for split in splits
    }


def source_frame_name(filename: PathLike) -> str:
    """Return the source-frame name of a (possibly augmented) image file.

    Roboflow names augmented copies ``<frame>_jpg.rf.<hash>.jpg``; every copy of
    one source frame shares the part before ``.rf.``. Names without that marker
    are returned without their suffix.
    """
    name = Path(filename).name
    if ".rf." in name:
        return name.split(".rf.")[0]
    return Path(name).stem


def list_images(directory: PathLike) -> List[Path]:
    """List image files in ``directory`` (sorted, by suffix, case-insensitive).

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image folder not found: {directory}")
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )


def find_split_overlap(
    root: PathLike, splits: Sequence[str] = SPLITS
) -> Dict[Tuple[str, str], Set[str]]:
    """Find source frames that appear in more than one split.

    Because training images were augmented into several copies, the same source
    frame appearing in two splits means near-identical images on both sides of
    the train/evaluation boundary (data leakage).

    Args:
        root: Dataset root containing ``<split>/images`` folders.
        splits: Split names to compare pairwise.

    Returns:
        Mapping from each pair of split names to the set of shared
        source-frame names (empty set when there is no overlap).
    """
    root = Path(root)
    frames = {
        split: {source_frame_name(p) for p in list_images(root / split / "images")}
        for split in splits
    }
    return {(a, b): frames[a] & frames[b] for a, b in combinations(splits, 2)}


def bbox_area_stats(
    label_dir: PathLike,
    class_names: Sequence[str] = CLASS_NAMES,
    skip_invalid: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Summarise annotation size per class.

    Size is the area of each annotation's axis-aligned bounding extent as a
    fraction of the image area (1.0 = whole image). Small values indicate
    objects that are hard to detect at reduced input resolution.

    Args:
        label_dir: Folder of ``.txt`` label files.
        class_names: Class names, indexed by class id.
        skip_invalid: Skip malformed lines instead of raising.

    Returns:
        ``{class_name: {"n", "median", "min", "max"}}`` for classes that occur;
        classes with no annotations are omitted.

    Raises:
        FileNotFoundError: If ``label_dir`` does not exist.
    """
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label folder not found: {label_dir}")

    areas: Dict[str, List[float]] = {}
    for label_file in sorted(label_dir.glob("*.txt")):
        for class_id, points in load_yolo_label(label_file, len(class_names), skip_invalid):
            width, height = _extent(points)
            areas.setdefault(class_names[class_id], []).append(width * height)

    return {
        name: {
            "n": len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        for name, values in areas.items()
    }


# --------------------------------------------------------------------------- #
# Extraction and Ultralytics config
# --------------------------------------------------------------------------- #
def _missing_split_dirs(root: Path, splits: Sequence[str]) -> List[str]:
    """Names of expected ``<split>/images`` and ``<split>/labels`` that are absent."""
    missing = []
    for split in splits:
        for sub in ("images", "labels"):
            if not (root / split / sub).is_dir():
                missing.append(f"{split}/{sub}")
    return missing


def extract_dataset(
    zip_path: PathLike,
    dest: PathLike,
    overwrite: bool = False,
    splits: Sequence[str] = SPLITS,
) -> Path:
    """Extract the dataset zip and check it has the expected layout.

    Args:
        zip_path: Path to the Roboflow YOLO export (``.zip``).
        dest: Folder to extract into.
        overwrite: If True, delete ``dest`` first if it already exists.
        splits: Split folders that must exist after extraction.

    Returns:
        ``dest`` as a :class:`pathlib.Path`.

    Raises:
        FileNotFoundError: If ``zip_path`` does not exist.
        zipfile.BadZipFile: If the file is not a valid zip.
        FileExistsError: If ``dest`` is a non-empty folder and ``overwrite`` is False.
        ValueError: If the extracted folder lacks any ``<split>/images`` or
            ``<split>/labels`` folder.
    """
    zip_path, dest = Path(zip_path), Path(dest)
    if not zip_path.is_file():
        raise FileNotFoundError(f"dataset zip not found: {zip_path}")

    if dest.exists() and any(dest.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{dest} already exists and is not empty; pass overwrite=True to replace it"
            )
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)

    missing = _missing_split_dirs(dest, splits)
    if missing:
        raise ValueError(
            f"unexpected dataset layout in {dest}; missing: {', '.join(missing)}"
        )
    return dest


def write_data_yaml(
    root: PathLike,
    class_names: Sequence[str] = CLASS_NAMES,
    out_path: Optional[PathLike] = None,
) -> Path:
    """Write an Ultralytics data config with absolute paths.

    The dataset's own ``data.yaml`` uses relative paths that only resolve from
    certain working directories; absolute paths avoid that ambiguity.

    Args:
        root: Dataset root containing ``train``, ``valid`` and ``test`` folders.
        class_names: Class names, indexed by class id.
        out_path: Where to write the file. Defaults to ``<root>/data_fixed.yaml``.

    Returns:
        Path of the written YAML file.

    Raises:
        ValueError: If any split's ``images``/``labels`` folder is missing.
    """
    root = Path(root).resolve()
    missing = _missing_split_dirs(root, SPLITS)
    if missing:
        raise ValueError(f"cannot write data yaml; missing in {root}: {', '.join(missing)}")

    out_path = Path(out_path) if out_path is not None else root / "data_fixed.yaml"
    # json.dumps gives valid YAML scalars/flow sequences and safely quotes
    # paths that contain spaces or colons.
    content = (
        f"train: {json.dumps(str(root / 'train' / 'images'))}\n"
        f"val: {json.dumps(str(root / 'valid' / 'images'))}\n"
        f"test: {json.dumps(str(root / 'test' / 'images'))}\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(list(class_names))}\n"
    )
    out_path.write_text(content)
    return out_path


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print dataset statistics: class counts, split overlap and box sizes."""
    parser = argparse.ArgumentParser(description="Summarise the pork rasher dataset.")
    parser.add_argument("--root", required=True, help="dataset root (contains train/valid/test)")
    args = parser.parse_args(argv)
    root = Path(args.root)

    summaries = class_counts_by_split(root)
    print("Instances per class")
    header = f"{'class':17s}" + "".join(f"{s:>8s}" for s in SPLITS) + f"{'total':>8s}"
    print(header)
    for name in CLASS_NAMES:
        row = [summaries[s].counts[name] for s in SPLITS]
        print(f"{name:17s}" + "".join(f"{v:8d}" for v in row) + f"{sum(row):8d}")
    for s in SPLITS:
        info = summaries[s]
        print(
            f"{s}: {info.n_label_files} images, {info.n_background} background, "
            f"{info.n_instances} instances"
        )

    print("\nSource frames shared between splits (possible leakage)")
    for (a, b), shared in find_split_overlap(root).items():
        print(f"  {a} & {b}: {len(shared)}")

    print("\nTraining annotation size (fraction of image area)")
    for name, stats in bbox_area_stats(root / "train" / "labels").items():
        print(
            f"  {name:17s} n={int(stats['n']):5d} median={stats['median']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )


if __name__ == "__main__":
    main()
