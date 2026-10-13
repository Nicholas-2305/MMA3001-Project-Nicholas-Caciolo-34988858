"""Tests for porkrasher.visualise (skipped if matplotlib/Pillow are missing)."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # no display needed
PIL_Image = pytest.importorskip("PIL.Image")
import matplotlib.pyplot as plt

from porkrasher import visualise


def make_image(path, size=(100, 80)):
    PIL_Image.new("RGB", size, color=(30, 30, 30)).save(path)


def test_draw_annotations_draws_one_patch_per_label(tmp_path):
    image, label = tmp_path / "a.png", tmp_path / "a.txt"
    make_image(image)
    label.write_text("1 0.5 0.5 0.2 0.1\n3 0.3 0.3 0.4 0.4\n")
    fig, ax = plt.subplots()
    drawn = visualise.draw_annotations(image, label, ax)
    assert drawn == 2
    assert len(ax.patches) == 2
    plt.close(fig)


def test_draw_annotations_without_label_file_draws_image_only(tmp_path):
    image = tmp_path / "a.png"
    make_image(image)
    fig, ax = plt.subplots()
    assert visualise.draw_annotations(image, tmp_path / "absent.txt", ax) == 0
    plt.close(fig)


def test_draw_annotations_missing_image_raises(tmp_path):
    fig, ax = plt.subplots()
    with pytest.raises(FileNotFoundError):
        visualise.draw_annotations(tmp_path / "absent.png", tmp_path / "a.txt", ax)
    plt.close(fig)


def test_plot_samples_is_reproducible_and_saves(tmp_path):
    for i in range(5):
        make_image(tmp_path / f"img{i}.png")
        (tmp_path / f"img{i}.txt").write_text("1 0.5 0.5 0.2 0.1\n")
    out = tmp_path.parent / (tmp_path.name + "_fig.png")  # outside the image folder
    fig = visualise.plot_samples(tmp_path, tmp_path, n=4, seed=1, out_path=out)
    assert out.is_file()
    titles_a = [ax.get_title() for ax in fig.axes]
    plt.close(fig)
    fig = visualise.plot_samples(tmp_path, tmp_path, n=4, seed=1)
    assert [ax.get_title() for ax in fig.axes] == titles_a
    plt.close(fig)


def test_plot_samples_rejects_empty_folder_and_bad_n(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        visualise.plot_samples(tmp_path, tmp_path)
    make_image(tmp_path / "a.png")
    with pytest.raises(ValueError, match="positive"):
        visualise.plot_samples(tmp_path, tmp_path, n=0)
