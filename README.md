# Anomaly Detection in Malaria Images

Data-preparation pipeline for the open-source **malaria cell dataset** (27,558 thin
blood-smear cell images, split evenly between *parasitized* and *uninfected*).
The project loads, explores, preprocesses, splits, and augments the images so they
are ready for a model.

## Project structure

```
anomaly-detection-in-malaria-images/
├── datasets/                       # downloaded dataset (created on first run)
├── requirements.txt                # Python dependencies (pinned lockfile)
├── tasks/
│   └── PREPROCESSING.md            # full pipeline reference (all steps in one doc)
├── src/
│   └── anomaly_detection/
│       ├── __init__.py             # suppresses noisy TensorFlow logs
│       ├── __main__.py             # entry point — runs the pipeline in order
│       ├── config.py               # paths, split ratios, BATCH_SIZE, VISUALIZE
│       └── data/
│           ├── load.py             # load the dataset
│           ├── visualize.py        # show sample images
│           ├── preprocess.py       # resize / normalize
│           ├── split.py            # train / val / test split
│           └── augment.py          # augment training data
└── README.md
```

## System / environment

Developed and tested on:

- **OS:** macOS
- **Python:** 3.9
- **Virtual environment:** `venv` (see setup below)

Other platforms should work too, as long as the dependencies in `requirement.txt`
install cleanly.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

The package lives inside `src/`, so Python needs to be told to look there. Set the
`PYTHONPATH` export first, then run the package:

```bash
export PYTHONPATH=src
python3 -m anomaly_detection
```

Or as a single line:

```bash
PYTHONPATH=src python3 -m anomaly_detection
```

> `PYTHONPATH=src` tells Python to also search the `src/` folder for packages, which
> is where `anomaly_detection` lives. Run these commands from the project root.

On the first run, the dataset (~337 MB) downloads into the `datasets/` folder. This
happens once; later runs read the cached data.

## Pipeline

The full step-by-step data-preparation workflow (import, load, visualize, preprocess,
split, augment) is documented in [`tasks/PREPROCESSING.md`](tasks/PREPROCESSING.md).

## Configuration

Edit `src/anomaly_detection/config.py`:

- `VISUALIZE` — set `True` to pop up the sample / preprocessing / augmentation image
  windows; set `False` to run the pipeline without any plots.
- `BATCH_SIZE` — how many images to process; `0` uses the full dataset, a positive
  number works on a smaller subset to run faster.
- `TRAIN_SET_SIZE`, `VAL_SET_SIZE`, `TEST_SIZE` — the split ratios (default 70/15/15).
- `DATA_DIR` — absolute path to the `datasets/` folder (computed automatically).

## Source

Based on the Educative project
[**Anomaly Detection in Medical Images (Python, TF & PyTorch)**](https://www.educative.io/projects/anomaly-detection-in-medical-images-python-tf-and-pytorch).
