# How the Data Pipeline Works

This document describes how this project turns the raw malaria dataset into clean,
split, and augmented data ready for a model. It reflects the actual implementation in
`src/anomaly_detection/`, written in my own words as a reference for the codebase.

**Dataset:** the open-source malaria dataset (via TensorFlow Datasets) — 27,558
blood-smear cell images, roughly half *parasitized* and half *uninfected*
(`label 0 = parasitized`, `label 1 = uninfected`).

The stages below are orchestrated by `prepare_data()` in `data/pipeline.py`, which
returns ready-to-use `(train, val, test)` splits. The app entry point (`__main__.py`)
calls it; `data/__main__.py` runs the same pipeline standalone for testing
(`python -m anomaly_detection.data`).

## Load — `data/load.py`

Fetches the dataset through TensorFlow Datasets and stores it under the project's
`datasets/` folder (path resolved from `config.DATA_DIR`, so it works regardless of
where the program is launched). Building the dataset reads only its metadata; the
image files download on the first `download_and_prepare()` call and are cached
afterwards.

## Visualize — `data/visualize.py`

Displays a handful of raw cell images with matplotlib. This is an eyeball check before
any transformation — it helps confirm the data loaded correctly and gives a feel for
what the cells look like. Runs only when `config.VISUALIZE` is enabled.

## Preprocess — `data/preprocess.py`

Two roles live here:

- `preprocess_and_visualize` is an exploratory view that shows a sample image through
  several candidate transforms (resize, grayscale, normalize, blur) side by side, so
  the effect of each is visible.
- `preprocess_image_dataset` is the real pipeline transform applied to every split:
  it resizes each image to 64x64 and scales pixel values into the 0-1 range. Keeping
  this identical across train, validation, and test avoids any scale mismatch.

## Split — `data/split.py`

Shuffles the data once (with a frozen order so the result is stable) and divides it
into training, validation, and test sets in a 70/15/15 ratio. The split uses
`take`/`skip` so the three sets never share an image — this prevents leakage, where
evaluation data sneaking into training would make the model look better than it really
is. The split ratios come from `config.py`, and the code prints the size and
class balance of each set as a sanity check.

## Augment — `data/augment.py`

Adds variety to the **training set only**, leaving validation and test untouched so
evaluation stays honest. Two augmentation styles are implemented:

- A Keras `ImageDataGenerator` used to preview random transforms (rotation, shift,
  shear, zoom, flips) on a single image — handy for seeing what augmentation does.
- `augment_image`, applied across the training split with `tf.data`'s `map`, which
  performs on-the-fly random flips, brightness, and contrast changes. Because the
  transforms are random and re-applied every pass over the data, the model effectively
  sees fresh variations of each image without storing extra copies.

## Orchestration — `data/pipeline.py`

`prepare_data(batch_size, malaria_builder=None)` ties the stages together: it loads
and downloads the dataset (if a builder isn't passed in), splits it, preprocesses
every split, augments the training set, and returns `(train, val, test)`. Keeping
this in a reusable function means both the app and the standalone data runner share
the exact same pipeline.

It is exposed as the data package's public API, so other code (e.g. the modeling
phase) can import it directly:

```python
from anomaly_detection.data import prepare_data

train, val, test = prepare_data(batch_size)
```

## Configuration — `config.py`

Central place for the knobs: `DATA_DIR` (dataset location), `VISUALIZE` (toggle the
plots), `BATCH_SIZE` (how much data to process; 0 = all), and the train/validation/test
split ratios.
