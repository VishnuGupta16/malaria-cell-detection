import tensorflow_datasets as tfds

from ..config import DATA_DIR


def load():
    """Task 2: Load the malaria dataset's info.

    tfds.builder reads the dataset's "description card" (name, size, classes).
    Note: this reads the info only — it does NOT download the 337 MB of images yet.
    Returns the builder so other steps can use it later.
    """
    malaria_builder = tfds.builder("malaria", data_dir=DATA_DIR)

    info = malaria_builder.info
    print(info.as_proto)
    print(info.splits)
    print(info.features)
    
    return malaria_builder


