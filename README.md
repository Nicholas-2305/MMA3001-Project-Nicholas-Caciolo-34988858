# Pork Rasher Packaging Defect Detection

MMA3001 individual project (Dataset 3: *Automated Detection of Pork Rasher Packaging Errors and Meat Quality*).

A YOLOv8 object detector, fine-tuned from COCO-pretrained weights, that finds packaging and meat-quality defects on pork rasher trays. Training was run in Google Colab (Tesla T4); this repository holds the reusable code, tests, documentation and results.

## What it does

- **Input:** a photo of a pork rasher tray (640 x 640 after resizing).
- **Output:** bounding boxes with a class and a confidence score for five classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.
- **Labels are plain bounding boxes** (no polygons), so this is detection, not segmentation.

## Repository layout

```
src/porkrasher/
    data.py        label parsing, class statistics, split-overlap check, dataset extraction, data.yaml
    train.py       fine-tune a YOLOv8 model (wraps ultralytics)
    evaluate.py    metrics on val/test, results tables, inference timing
    visualise.py   draw ground-truth boxes for sanity checks
tests/             pytest tests (no GPU, no dataset and no ultralytics needed)
notebooks/         colab_workflow.ipynb: thin notebook that calls the package
results/           metrics_summary.csv, per_class_ap50.csv
docs/              generated HTML documentation
```

## Setup

```bash
git clone <https://github.com/Nicholas-2305/MMA3001-Project-Nicholas-Caciolo-34988858/tree/main>
cd pork-rasher-defect-detection
pip install -e .            # installs ultralytics==8.4.175 and the other dependencies
pip install pytest pdoc     # development tools
```

## Data

The dataset is **not** included in this repository. It is the Roboflow Universe project
[pork-rasher-error-packaging, version 4](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4)
(workspace `hello-2aqe0`), licensed **CC BY 4.0**. Download it in YOLOv8 format and unzip it so that the root contains `train/`, `valid/` and `test/`, each with `images/` and `labels/`.

| Split | Images | Instances |
|---|---|---|
| train | 3,349 | 2,923 |
| valid | 120 | 121 |
| test | 80 | 79 |

Class instances (train / valid / test): unsealed 1,055 / 81 / 49; packaging-error 1,712 / 34 / 23; twisted-meat 72 / 3 / 4; loose-meat 66 / 3 / 3; wrinkle 18 / 0 / 0.

## Reproducing the results

```bash
# 1. inspect the data (class counts, split overlap, box sizes), then write the Ultralytics config
python -m porkrasher.data --root path/to/dataset
python -c "from porkrasher.data import write_data_yaml; print(write_data_yaml('path/to/dataset'))"

# 2. train (selected model: YOLOv8n; comparison model: YOLOv8s)
python -m porkrasher.train --data path/to/dataset/data.yaml --weights yolov8n.pt --epochs 100 --name yolov8n_100ep
python -m porkrasher.train --data path/to/dataset/data.yaml --weights yolov8s.pt --epochs 100 --name yolov8s_100ep

# 3. validate on the validation split (use this for model selection)
python -m porkrasher.evaluate --weights runs/yolov8n_100ep/weights/best.pt --data path/to/dataset/data.yaml --split val --out results/yolov8n_val.json

# 4. test split: run ONCE, on the final chosen model only
python -m porkrasher.evaluate --weights runs/yolov8n_100ep/weights/best.pt --data path/to/dataset/data.yaml --split test --out results/yolov8n_test.json
```

`notebooks/colab_workflow.ipynb` runs the same steps in Colab. Set `REPO_URL` and `ZIP_PATH` in its first cells.

## Results

| Model | Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| YOLOv8n (50 epochs) | val | 0.606 | 0.360 | 0.781 | 0.559 |
| **YOLOv8n (100 epochs, selected)** | val | 0.617 | 0.377 | 0.753 | 0.568 |
| **YOLOv8n (100 epochs, selected)** | test | 0.512 | 0.247 | 0.569 | 0.453 |
| YOLOv8s (100 epochs) | val | 0.587 | 0.368 | 0.531 | 0.599 |
| YOLOv8s (100 epochs) | test | 0.498 | 0.239 | 0.526 | 0.522 |

Per-class AP50 is in `results/per_class_ap50.csv`.

**How to read these numbers**

- The two models are essentially tied on accuracy (0.014 mAP50 apart on test, 0.03 on validation). YOLOv8n was chosen because it is cheaper: 3.0 M vs 11.1 M parameters (27%), 8.1 vs 28.4 GFLOPs (29%), and faster inference in the timings taken.
- This is a **single-seed** comparison on small evaluation sets (120 validation and 80 test images), so small differences are not reliable.
- The YOLOv8s test run was done after the model was selected and in a different Colab session. It is supplementary, not part of the selection.
- Inference timings come from different Colab sessions and are noisy.

## Limitations

- `wrinkle` has 18 instances in total and none in validation or test, so it is not evaluated. `twisted-meat` and `loose-meat` have only 3 to 4 instances per evaluation split, so their scores are highly unstable (twisted-meat AP50 is 0.665 on validation but 0.03 on test).
- The training set contains about 1,085 distinct source frames augmented to 3,349 images. A few source frames appear in more than one split (train and valid: 8, train and test: 4, valid and test: 1), and some frames are probably near-duplicates of neighbouring frames. Validation and test scores may therefore be optimistic. Check with `porkrasher.data.find_split_overlap`.
- `packaging-error` has the most training instances but only a middling AP50. Its boxes are tiny (median 0.36% of the image area), which may be the cause; this was not tested.

## Tests and documentation

```bash
pytest -v                          # run the tests
pytest -v > results/test_report.txt   # save a test report
pdoc -o docs src/porkrasher        # regenerate the HTML documentation
```

Open `docs/index.html` in a browser.

## AI use

Claude (Anthropic) was used to help plan the project, write and debug code, draft the report and structure this repository. All results were produced by running the code on the dataset; see the AI-use reflection in the report.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, from Roboflow Universe (workspace `hello-2aqe0`); it is separate from this repository's license.
