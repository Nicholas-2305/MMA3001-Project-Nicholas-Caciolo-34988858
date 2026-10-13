"""Draw ground-truth annotations on images for sanity checking.

Run this before training: boxes that land in the wrong place reveal path,
format or label problems early (this is how the project confirmed the labels
were plain boxes, not polygons). The figures also serve as report figures.

``matplotlib`` and ``Pillow`` are imported when a function is called.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Union

from porkrasher.data import CLASS_NAMES, list_images, load_yolo_label

PathLike = Union[str, Path]


def draw_annotations(
    image_path: PathLike,
    label_path: PathLike,
    ax,
    class_names: Sequence[str] = CLASS_NAMES,
) -> int:
    """Draw one image and its annotations on a matplotlib axis.

    Boxes and polygons are drawn identically (see
    :func:`porkrasher.data.parse_label_line`).

    Args:
        image_path: Image file.
        label_path: Matching YOLO label file. If it does not exist, the image
            is drawn with no annotations (a background image).
        ax: A ``matplotlib.axes.Axes`` to draw on.
        class_names: Class names, indexed by class id.

    Returns:
        Number of annotations drawn.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If the label file contains a malformed line.
    """
    import matplotlib.patches as patches
    from PIL import Image

    image_path, label_path = Path(image_path), Path(label_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size
        ax.imshow(img)

    annotations = (
        load_yolo_label(label_path, len(class_names)) if label_path.is_file() else []
    )
    for class_id, points in annotations:
        pixels = [(x * width, y * height) for x, y in points]
        ax.add_patch(
            patches.Polygon(pixels, closed=True, fill=False, edgecolor="red", linewidth=1.5)
        )
        ax.text(
            pixels[0][0],
            pixels[0][1],
            class_names[class_id],
            color="yellow",
            fontsize=8,
            backgroundcolor="black",
        )
    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")
    return len(annotations)


def plot_samples(
    image_dir: PathLike,
    label_dir: PathLike,
    n: int = 4,
    seed: int = 0,
    out_path: Optional[PathLike] = None,
):
    """Plot a random sample of annotated images in a grid.

    Args:
        image_dir: Folder of images.
        label_dir: Folder of matching ``.txt`` labels.
        n: Number of images to show (capped at the number available).
        seed: Seed for the random sample, so figures are reproducible.
        out_path: If given, save the figure here.

    Returns:
        The ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If ``n`` is not positive or the folder has no images.
        FileNotFoundError: If ``image_dir`` does not exist.
    """
    import matplotlib.pyplot as plt

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    images = list_images(image_dir)
    if not images:
        raise ValueError(f"no images found in {image_dir}")

    chosen = random.Random(seed).sample(images, min(n, len(images)))
    cols = 2 if len(chosen) > 1 else 1
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    for ax, image_path in zip(axes.flat, chosen):
        label_path = Path(label_dir) / (image_path.stem + ".txt")
        draw_annotations(image_path, label_path, ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig
