"""Shared test fixtures: a tiny fake dataset with known contents."""

import pytest


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def fake_dataset(tmp_path):
    """Build a miniature YOLO dataset and return its root folder.

    Contents (class ids: 1 = packaging-error, 3 = unsealed, 0 = loose-meat):

    * train: ``frame_a`` has two augmented copies (h1, h2); ``frame_b`` is a
      background image (empty label file).
    * valid: ``frame_a`` again (deliberate leakage) and ``frame_c``.
    * test: ``frame_d``.
    """
    root = tmp_path / "dataset"
    layout = {
        "train": {
            "frame_a_jpg.rf.h1": "1 0.5 0.5 0.2 0.1\n3 0.3 0.3 0.4 0.4\n",
            "frame_a_jpg.rf.h2": "1 0.5 0.5 0.2 0.1\n",
            "frame_b_jpg.rf.h3": "",
        },
        "valid": {
            "frame_a_jpg.rf.h9": "3 0.5 0.5 0.5 0.5\n",
            "frame_c_jpg.rf.h8": "0 0.5 0.5 0.3 0.3\n",
        },
        "test": {
            "frame_d_jpg.rf.h7": "1 0.1 0.1 0.1 0.1\n",
        },
    }
    for split, items in layout.items():
        for stem, label_text in items.items():
            _write(root / split / "images" / f"{stem}.jpg", "fake image bytes")
            _write(root / split / "labels" / f"{stem}.txt", label_text)
    return root
