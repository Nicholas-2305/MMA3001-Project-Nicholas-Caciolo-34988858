"""Train a YOLOv8 object detector on the pork rasher dataset.

Wraps ``ultralytics.YOLO.train`` so every run is described by explicit
arguments and leaves a ``train_summary.json`` beside its weights. The summary
records the settings, the wall-clock training time, and the *actual* output
folder, because Ultralytics silently renames a run folder if the name already
exists (``name``, ``name2``, ...). Always use the path this module prints or
returns rather than assuming the folder name you passed in.

Example (the three runs reported in the project)::

    python -m porkrasher.train --data data/dataset/data_fixed.yaml \\
        --weights yolov8n.pt --epochs 50  --name nano_50ep
    python -m porkrasher.train --data data/dataset/data_fixed.yaml \\
        --weights yolov8n.pt --epochs 100 --name nano_100ep
    python -m porkrasher.train --data data/dataset/data_fixed.yaml \\
        --weights yolov8s.pt --epochs 100 --name small_100ep

``ultralytics`` is imported only when training starts, so argument validation
(and the unit tests) work without it installed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]


def train(
    weights: str,
    data_yaml: PathLike,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 15,
    project: PathLike = "runs",
    name: str = "run",
    seed: int = 0,
) -> Path:
    """Fine-tune a pretrained YOLOv8 detector and return the best weights path.

    Args:
        weights: Pretrained weights name or path, e.g. ``"yolov8n.pt"``
            (nano) or ``"yolov8s.pt"`` (small). Names are downloaded by
            Ultralytics on first use. Must be a *detection* model, not
            ``-seg``: this dataset only has bounding-box labels.
        data_yaml: Ultralytics data config (see
            :func:`porkrasher.data.write_data_yaml`).
        epochs: Maximum number of epochs; training may stop earlier.
        imgsz: Square input resolution in pixels (images are letterboxed).
        batch: Batch size. 16 fits a free-tier Colab T4 for the models used.
        patience: Stop if validation fitness has not improved for this many
            epochs.
        project: Parent folder for runs.
        name: Run name; Ultralytics may append a number if it already exists.
        seed: Random seed.

    Returns:
        Path to ``weights/best.pt`` inside the run folder that was actually
        created.

    Raises:
        ValueError: If a numeric argument is not positive, or ``weights`` is a
            segmentation model.
        FileNotFoundError: If ``data_yaml`` does not exist.
    """
    # Validate cheap arguments first so a typo fails in milliseconds, not after
    # the heavy ultralytics import and a model download.
    for label, value in (
        ("epochs", epochs),
        ("imgsz", imgsz),
        ("batch", batch),
        ("patience", patience),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
    # Segmentation weights train a mask head and raise an obscure error on
    # box-only labels; catch the mistake here with a clear message instead.
    if str(weights).lower().endswith("-seg.pt"):
        raise ValueError(
            "segmentation weights given, but the dataset only has bounding-box "
            "labels; use a detection model such as yolov8n.pt"
        )
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data config not found: {data_yaml}")

    from ultralytics import YOLO  # imported lazily: heavy, and not needed to validate args

    started = time.perf_counter()
    model = YOLO(weights)  # COCO-pretrained: transfer learning, not training from scratch
    # exist_ok=False makes Ultralytics create a new numbered folder rather than
    # overwrite an earlier run, so earlier results are never lost.
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(project),
        name=name,
        seed=seed,
        exist_ok=False,
    )
    elapsed_hours = (time.perf_counter() - started) / 3600

    # Ask the trainer where it actually saved. Guessing project/name is wrong
    # whenever a name is reused, because Ultralytics then appends a number
    # (name2, name3, ...) and the guess points at an older run's weights.
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    import ultralytics

    # Record the settings next to the weights so every result can be traced to
    # the exact configuration and library version that produced it.
    summary = {
        "weights": str(weights),
        "data": str(data_yaml),
        "epochs_max": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "seed": seed,
        "elapsed_hours": round(elapsed_hours, 3),
        "ultralytics_version": ultralytics.__version__,
        "save_dir": str(save_dir),
        "best_weights": str(best),
    }
    (save_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Run folder (use this path): {save_dir}")
    print(f"Training took {elapsed_hours:.2f} h; epochs actually run are in results.csv")
    return best


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point; see the module docstring for examples."""
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector.")
    parser.add_argument("--data", required=True, help="Ultralytics data yaml")
    parser.add_argument("--weights", default="yolov8n.pt", help="pretrained weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    train(
        weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
