"""Tests for porkrasher.data."""

import zipfile

import pytest

from porkrasher import data


# ----------------------------- label parsing ------------------------------ #
def test_parse_box_returns_four_corners():
    class_id, points = data.parse_label_line("1 0.5 0.5 0.2 0.1")
    assert class_id == 1
    assert points == [
        pytest.approx((0.4, 0.45)),
        pytest.approx((0.6, 0.45)),
        pytest.approx((0.6, 0.55)),
        pytest.approx((0.4, 0.55)),
    ]


def test_parse_polygon_keeps_vertices():
    class_id, points = data.parse_label_line("3 0.1 0.1 0.9 0.1 0.5 0.8")
    assert class_id == 3
    assert points == [(0.1, 0.1), (0.9, 0.1), (0.5, 0.8)]


def test_parse_blank_line_returns_none():
    assert data.parse_label_line("   \n") is None


@pytest.mark.parametrize(
    "bad_line, message",
    [
        ("9 0.5 0.5 0.2 0.1", "out of range"),  # class id too large
        ("-1 0.5 0.5 0.2 0.1", "out of range"),  # negative class id
        ("a 0.5 0.5 0.2 0.1", "non-numeric"),  # class id not a number
        ("1 0.5 x 0.2 0.1", "non-numeric"),  # coordinate not a number
        ("1 0.5 0.5 0.2", "expected 4 values"),  # 3 coordinates
        ("1 0.5 0.5 0.2 0.1 0.3", "expected 4 values"),  # 5 coordinates
        ("1 0.1 0.1 0.2 0.2 0.3 0.3 0.4", "expected 4 values"),  # odd polygon
        ("1 0.5 0.5 1.5 0.1", "outside"),  # coordinate > 1
    ],
)
def test_parse_rejects_malformed_lines(bad_line, message):
    with pytest.raises(ValueError, match=message):
        data.parse_label_line(bad_line)


def test_load_label_error_names_file_and_line(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("1 0.5 0.5 0.2 0.1\n9 0.5 0.5 0.2 0.1\n")
    with pytest.raises(ValueError, match=r"x\.txt:2"):
        data.load_yolo_label(label)


def test_load_label_skip_invalid_keeps_good_lines(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("1 0.5 0.5 0.2 0.1\nnonsense\n3 0.5 0.5 0.2 0.1\n")
    annotations = data.load_yolo_label(label, skip_invalid=True)
    assert [class_id for class_id, _ in annotations] == [1, 3]


def test_load_label_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_yolo_label(tmp_path / "absent.txt")


def test_load_label_empty_file_is_background(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("")
    assert data.load_yolo_label(label) == []


# ------------------------------ statistics -------------------------------- #
def test_summarise_split_counts_classes_and_background(fake_dataset):
    summary = data.summarise_split(fake_dataset / "train" / "labels")
    assert summary.n_label_files == 3
    assert summary.n_background == 1
    assert summary.counts["packaging-error"] == 2
    assert summary.counts["unsealed"] == 1
    assert summary.counts["wrinkle"] == 0  # Counter returns 0 for unseen classes
    assert summary.n_instances == 3


def test_summarise_split_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.summarise_split(tmp_path / "nope")


def test_summarise_split_strict_vs_skip_invalid(tmp_path):
    (tmp_path / "a.txt").write_text("1 0.5 0.5 0.2 0.1\n")
    (tmp_path / "b.txt").write_text("9 0.5 0.5 0.2 0.1\n")
    with pytest.raises(ValueError):
        data.summarise_split(tmp_path)
    summary = data.summarise_split(tmp_path, skip_invalid=True)
    assert summary.counts["packaging-error"] == 1


def test_class_counts_by_split_covers_all_splits(fake_dataset):
    summaries = data.class_counts_by_split(fake_dataset)
    assert set(summaries) == {"train", "valid", "test"}
    assert summaries["valid"].counts["loose-meat"] == 1
    assert summaries["test"].counts["packaging-error"] == 1


def test_source_frame_name_strips_roboflow_hash():
    assert data.source_frame_name("frame_with_red_432_jpg.rf.8be4a0.jpg") == "frame_with_red_432_jpg"
    assert data.source_frame_name("plain_image.png") == "plain_image"


def test_list_images_filters_and_sorts(tmp_path):
    for name in ("b.jpg", "a.PNG", "notes.txt", "c.jpeg"):
        (tmp_path / name).write_text("x")
    assert [p.name for p in data.list_images(tmp_path)] == ["a.PNG", "b.jpg", "c.jpeg"]


def test_list_images_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.list_images(tmp_path / "nope")


def test_find_split_overlap_detects_shared_frame(fake_dataset):
    overlap = data.find_split_overlap(fake_dataset)
    assert overlap[("train", "valid")] == {"frame_a_jpg"}
    assert overlap[("train", "test")] == set()
    assert overlap[("valid", "test")] == set()


def test_bbox_area_stats_reports_fraction_of_image(fake_dataset):
    stats = data.bbox_area_stats(fake_dataset / "train" / "labels")
    assert stats["packaging-error"]["n"] == 2
    assert stats["packaging-error"]["median"] == pytest.approx(0.2 * 0.1)
    assert stats["unsealed"]["median"] == pytest.approx(0.4 * 0.4)
    assert "wrinkle" not in stats  # no instances, so omitted


def test_bbox_area_stats_uses_polygon_extent(tmp_path):
    (tmp_path / "a.txt").write_text("1 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n")
    stats = data.bbox_area_stats(tmp_path)
    assert stats["packaging-error"]["median"] == pytest.approx(0.4 * 0.2)


# --------------------------- extraction / config -------------------------- #
def _zip_dataset(root, zip_path):
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root))


def test_extract_dataset_round_trip(fake_dataset, tmp_path):
    zip_path = tmp_path / "data.zip"
    _zip_dataset(fake_dataset, zip_path)
    out = data.extract_dataset(zip_path, tmp_path / "out")
    assert (out / "train" / "labels").is_dir()
    assert data.summarise_split(out / "train" / "labels").n_label_files == 3


def test_extract_dataset_missing_zip_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.extract_dataset(tmp_path / "absent.zip", tmp_path / "out")


def test_extract_dataset_rejects_wrong_layout(tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("readme.txt", "not a dataset")
    with pytest.raises(ValueError, match="unexpected dataset layout"):
        data.extract_dataset(zip_path, tmp_path / "out")


def test_extract_dataset_refuses_nonempty_dest_unless_overwrite(fake_dataset, tmp_path):
    zip_path = tmp_path / "data.zip"
    _zip_dataset(fake_dataset, zip_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "stale.txt").write_text("old")
    with pytest.raises(FileExistsError):
        data.extract_dataset(zip_path, dest)
    data.extract_dataset(zip_path, dest, overwrite=True)
    assert not (dest / "stale.txt").exists()


def test_write_data_yaml_contains_absolute_paths_and_classes(fake_dataset):
    out = data.write_data_yaml(fake_dataset)
    text = out.read_text()
    assert str((fake_dataset / "train" / "images").resolve()) in text
    assert "nc: 5" in text
    assert '"wrinkle"' in text
    assert "val:" in text and "test:" in text


def test_write_data_yaml_requires_all_splits(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    with pytest.raises(ValueError, match="missing"):
        data.write_data_yaml(tmp_path)
