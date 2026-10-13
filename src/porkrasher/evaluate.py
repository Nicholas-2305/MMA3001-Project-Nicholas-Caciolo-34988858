"""Evaluate trained detectors and summarise results.

Typical use::

    # validation split: use freely while comparing models
    python -m porkrasher.evaluate --weights runs/nano_100ep/weights/best.pt \\
        --data data/dataset/data_fixed.yaml --split val --out results/nano_val.json

    # test split: run ONCE, on the single model chosen using validation results
    python -m porkrasher.evaluate --weights runs/nano_100ep/weights/best.pt \\
        --data data/dataset/data_fixed.yaml --split test --out results/nano_test.json

The test split exists to give an estimate that no modelling decision has
touched. Evaluating several models on it and then picking the best turns it
into a second validation set.

Classes with no instances in the evaluated split (``wrinkle`` in this dataset)
cannot be scored and are omitted from per-class results, not reported as 0.

``ultralytics`` is imported only when a model is actually evaluated.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Splits that ``ultralytics`` can evaluate here.
VALID_SPLITS = ("val", "test")


def metrics_to_dict(
    metrics: Any, class_names: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Convert an Ultralytics detection-metrics object into plain Python data.

    Args:
        metrics: Object returned by ``YOLO.val`` (anything exposing ``.box``
            with ``map50``, ``map``, ``mp``, ``mr``, ``ap_class_index``, ``p``,
            ``r``, ``ap50``, ``ap``; optionally ``.names`` and ``.speed``).
        class_names: Fallback class names (by id) if ``metrics.names`` is absent.

    Returns:
        ``{"overall": {...}, "per_class": {name: {...}}, "speed_ms": {...}}``
        where metric keys are ``map50``, ``map50_95``, ``precision`` and
        ``recall``. Only classes present in the evaluated split appear in
        ``per_class``.
    """
    box = metrics.box
    names = getattr(metrics, "names", None)
    if names is None and class_names is not None:
        names = {i: n for i, n in enumerate(class_names)}
    if names is None:
        names = {}

    per_class: Dict[str, Dict[str, float]] = {}
    for position, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        per_class[str(names.get(class_id, class_id))] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "map50": float(box.ap50[position]),
            "map50_95": float(box.ap[position]),
        }

    speed = getattr(metrics, "speed", None) or {}
    return {
        "overall": {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "per_class": per_class,
        "speed_ms": {k: float(v) for k, v in speed.items()},
    }


def format_results_table(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Render overall metrics for several models as a Markdown table.

    Args:
        results: ``{model_label: metrics_dict}`` where each value comes from
            :func:`metrics_to_dict` (or :func:`evaluate`).

    Returns:
        Markdown table text with one row per model.
    """
    lines = [
        "| Model | mAP50 | mAP50-95 | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for label, result in results.items():
        o = result["overall"]
        lines.append(
            f"| {label} | {o['map50']:.3f} | {o['map50_95']:.3f} "
            f"| {o['precision']:.3f} | {o['recall']:.3f} |"
        )
    return "\n".join(lines)


def format_per_class_table(
    results: Mapping[str, Mapping[str, Any]], metric: str = "map50"
) -> str:
    """Render one per-class metric for several models as a Markdown table.

    Args:
        results: ``{model_label: metrics_dict}``.
        metric: One of ``"map50"``, ``"map50_95"``, ``"precision"``, ``"recall"``.

    Returns:
        Markdown table with one row per class (union across models, sorted by
        name). A class missing from a model's results shows an em dash.

    Raises:
        ValueError: If ``metric`` is not a recognised key.
    """
    if metric not in ("map50", "map50_95", "precision", "recall"):
        raise ValueError(f"unknown metric: {metric!r}")
    labels = list(results)
    classes = sorted({c for r in results.values() for c in r["per_class"]})
    lines = [
        "| Class | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for cls in classes:
        cells = []
        for label in labels:
            entry = results[label]["per_class"].get(cls)
            cells.append("—" if entry is None else f"{entry[metric]:.3f}")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def evaluate(
    weights: PathLike,
    data_yaml: PathLike,
    split: str = "val",
    imgsz: int = 640,
    out_json: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Evaluate trained weights on the validation or test split.

    Args:
        weights: Path to a trained ``best.pt``.
        data_yaml: Ultralytics data config.
        split: ``"val"`` or ``"test"``. Use ``"test"`` once, on the final model.
        imgsz: Square input resolution; should match training.
        out_json: If given, write the result dictionary to this JSON file.

    Returns:
        The dictionary from :func:`metrics_to_dict`, plus ``"split"``,
        ``"weights"`` and ``"data"`` keys.

    Raises:
        ValueError: If ``split`` is not ``"val"`` or ``"test"``.
        FileNotFoundError: If ``weights`` or ``data_yaml`` does not exist.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    weights, data_yaml = Path(weights), Path(data_yaml)
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data config not found: {data_yaml}")
    if split == "test":
        logger.warning(
            "Evaluating on the TEST split: run this once, on the model chosen "
            "using validation results only."
        )

    from ultralytics import YOLO  # lazy: heavy dependency

    metrics = YOLO(str(weights)).val(data=str(data_yaml), split=split, imgsz=imgsz)
    result = metrics_to_dict(metrics)
    result.update({"split": split, "weights": str(weights), "data": str(data_yaml)})
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=2))
    return result


def benchmark_inference(
    weights: PathLike,
    image_paths: Sequence[PathLike],
    imgsz: int = 640,
    conf: float = 0.25,
    n_warmup: int = 10,
) -> Dict[str, float]:
    """Time end-to-end inference (load, resize, forward pass, NMS) per image.

    Warm-up predictions are run first and excluded, so one-off GPU start-up
    cost is not counted. Compare models only when benchmarked in the *same*
    session on the *same* images: timings vary between Colab sessions.

    Args:
        weights: Path to trained weights.
        image_paths: Images to time (one prediction each).
        imgsz: Square input resolution.
        conf: Confidence threshold.
        n_warmup: Number of untimed warm-up predictions.

    Returns:
        ``{"ms_per_image": float, "n_images": int}``.

    Raises:
        ValueError: If ``image_paths`` is empty or ``n_warmup`` is negative.
        FileNotFoundError: If ``weights`` does not exist.
    """
    images = [str(p) for p in image_paths]
    if not images:
        raise ValueError("image_paths is empty")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
    if not Path(weights).is_file():
        raise FileNotFoundError(f"weights not found: {weights}")

    from ultralytics import YOLO  # lazy: heavy dependency

    model = YOLO(str(weights))
    for path in images[:n_warmup]:
        model.predict(path, imgsz=imgsz, conf=conf, verbose=False)

    started = time.perf_counter()
    for path in images:
        model.predict(path, imgsz=imgsz, conf=conf, verbose=False)
    elapsed = time.perf_counter() - started
    return {"ms_per_image": elapsed / len(images) * 1000, "n_images": len(images)}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point; see the module docstring for examples."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate a trained detector.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Ultralytics data yaml")
    parser.add_argument("--split", choices=VALID_SPLITS, default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", help="write results to this JSON file")
    parser.add_argument(
        "--benchmark-dir",
        help="also time inference over the images in this folder",
    )
    args = parser.parse_args(argv)

    result = evaluate(args.weights, args.data, args.split, args.imgsz, args.out)
    print(format_results_table({Path(args.weights).parent.parent.name: result}))
    print()
    print(format_per_class_table({Path(args.weights).parent.parent.name: result}))

    if args.benchmark_dir:
        from porkrasher.data import list_images

        timing = benchmark_inference(
            args.weights, list_images(args.benchmark_dir), imgsz=args.imgsz
        )
        print(f"\nInference: {timing['ms_per_image']:.1f} ms/image over {timing['n_images']} images")


if __name__ == "__main__":
    main()
