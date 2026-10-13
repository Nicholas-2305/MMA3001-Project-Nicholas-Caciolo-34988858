"""Pork rasher packaging-defect detection (MMA3001 individual project).

Modules
-------
``porkrasher.data``
    Dataset handling: extraction, YOLO label parsing, class counts, split-overlap
    and box-size checks.
``porkrasher.train``
    Train a YOLOv8 detector (command line or function call).
``porkrasher.evaluate``
    Evaluate trained weights on the validation or test split, summarise metrics,
    and time inference.
``porkrasher.visualise``
    Draw ground-truth annotations on images for sanity checking.

The submodules are deliberately not imported here, so that
``python -m porkrasher.data`` and friends run without import warnings and the
lightweight modules work without the heavy ``ultralytics`` dependency.
"""

__version__ = "0.1.0"
