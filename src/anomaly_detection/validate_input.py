import tensorflow as tf
from .config import MODEL_INPUT_SHAPE, CLASSES


def validate_sample(train_set, validation_batch_size=1):
    """Verify the prepared data before modeling (Task 7).

    Scans `validation_batch_size` samples from the training set and checks:
      - every image has the expected shape (MODEL_INPUT_SHAPE), and
      - the expected number of distinct classes shows up (CLASSES).
    """
    labels = set()
    checked = 0
    for sample in train_set.take(validation_batch_size):
        shape = sample["image"].shape          # one image -> (64, 64, 3); no [0]
        if shape != MODEL_INPUT_SHAPE:
            return f"SHAPE IS DIFFERENT: {shape}"
        labels.add(int(sample["label"]))       # collect unique labels across samples
        checked += 1

    print(f"Checked {checked} samples | image shape: {MODEL_INPUT_SHAPE} | classes found: {len(labels)}")

    if len(labels) != CLASSES:
        return "Classes are not same"

    return "Valid Sample"
