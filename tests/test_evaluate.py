"""Tests for porkrasher.evaluate and the argument validation in porkrasher.train.

These tests never import ultralytics: every function validates its arguments
before importing it, and metrics are summarised from a small stand-in object.
"""

from types import SimpleNamespace

import pytest

from porkrasher import evaluate, train


def make_metrics(with_speed=True):
    """Stand-in for Ultralytics detection metrics; wrinkle (id 4) is absent."""
    box = SimpleNamespace(
        map50=0.6,
        map=0.35,
        mp=0.7,
        mr=0.5,
        ap_class_index=[1, 3],  # packaging-error and unsealed are present
        p=[0.6, 0.9],
        r=[0.4, 0.8],
        ap50=[0.45, 0.95],
        ap=[0.15, 0.65],
    )
    names = {0: "loose-meat", 1: "packaging-error", 2: "twisted-meat", 3: "unsealed", 4: "wrinkle"}
    speed = {"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0} if with_speed else None
    return SimpleNamespace(box=box, names=names, speed=speed)


def test_metrics_to_dict_overall_and_per_class():
    result = evaluate.metrics_to_dict(make_metrics())
    assert result["overall"] == {
        "map50": 0.6,
        "map50_95": 0.35,
        "precision": 0.7,
        "recall": 0.5,
    }
    assert result["per_class"]["packaging-error"]["map50"] == pytest.approx(0.45)
    assert result["per_class"]["unsealed"]["recall"] == pytest.approx(0.8)
    assert result["speed_ms"]["inference"] == 5.0


def test_metrics_to_dict_omits_classes_with_no_instances():
    result = evaluate.metrics_to_dict(make_metrics())
    assert "wrinkle" not in result["per_class"]
    assert set(result["per_class"]) == {"packaging-error", "unsealed"}


def test_metrics_to_dict_handles_missing_speed():
    assert evaluate.metrics_to_dict(make_metrics(with_speed=False))["speed_ms"] == {}


def test_metrics_to_dict_values_are_plain_floats():
    result = evaluate.metrics_to_dict(make_metrics())
    assert all(type(v) is float for v in result["overall"].values())


def test_format_results_table_has_one_row_per_model():
    results = {
        "nano": evaluate.metrics_to_dict(make_metrics()),
        "small": evaluate.metrics_to_dict(make_metrics()),
    }
    lines = evaluate.format_results_table(results).splitlines()
    assert lines[0].startswith("| Model |")
    assert len(lines) == 4  # header, separator, two models
    assert "| nano | 0.600 | 0.350 | 0.700 | 0.500 |" in lines


def test_format_per_class_table_marks_missing_classes():
    a = evaluate.metrics_to_dict(make_metrics())
    b = evaluate.metrics_to_dict(make_metrics())
    del b["per_class"]["unsealed"]
    table = evaluate.format_per_class_table({"a": a, "b": b}, metric="map50")
    unsealed_row = [l for l in table.splitlines() if l.startswith("| unsealed")][0]
    assert unsealed_row == "| unsealed | 0.950 | — |"


def test_format_per_class_table_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate.format_per_class_table({"a": evaluate.metrics_to_dict(make_metrics())}, "f1")


# ------------------------- argument validation ---------------------------- #
def test_evaluate_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split"):
        evaluate.evaluate(tmp_path / "w.pt", tmp_path / "d.yaml", split="train")


def test_evaluate_missing_weights_raises(tmp_path):
    (tmp_path / "d.yaml").write_text("x")
    with pytest.raises(FileNotFoundError, match="weights"):
        evaluate.evaluate(tmp_path / "w.pt", tmp_path / "d.yaml")


def test_evaluate_missing_data_config_raises(tmp_path):
    (tmp_path / "w.pt").write_text("x")
    with pytest.raises(FileNotFoundError, match="data config"):
        evaluate.evaluate(tmp_path / "w.pt", tmp_path / "d.yaml")


def test_benchmark_rejects_empty_image_list(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        evaluate.benchmark_inference(tmp_path / "w.pt", [])


def test_benchmark_rejects_negative_warmup(tmp_path):
    with pytest.raises(ValueError, match="n_warmup"):
        evaluate.benchmark_inference(tmp_path / "w.pt", ["a.jpg"], n_warmup=-1)


@pytest.mark.parametrize("field", ["epochs", "imgsz", "batch", "patience"])
def test_train_rejects_non_positive_numbers(tmp_path, field):
    (tmp_path / "d.yaml").write_text("x")
    kwargs = {"weights": "yolov8n.pt", "data_yaml": tmp_path / "d.yaml", field: 0}
    with pytest.raises(ValueError, match=field):
        train.train(**kwargs)


def test_train_rejects_segmentation_weights(tmp_path):
    (tmp_path / "d.yaml").write_text("x")
    with pytest.raises(ValueError, match="segmentation"):
        train.train("yolov8n-seg.pt", tmp_path / "d.yaml")


def test_train_missing_data_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="data config"):
        train.train("yolov8n.pt", tmp_path / "absent.yaml")
