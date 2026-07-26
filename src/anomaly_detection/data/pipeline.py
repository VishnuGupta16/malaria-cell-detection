from .split import split
from .preprocess import preprocess_image_dataset
from .augment import augment
from .load import load


def prepare_data(batch_size, malaria_builder = None):

    """Run the full data pipeline and return ready-to-use splits.
  
      Steps:
        1. Split into train / validation / test.
        2. Preprocess every split the same way (resize + normalize).
        3. Augment the training set only.
  
      Returns (train, val, test). Reusable: call it from the app entry point
      or from data/__main__.py for standalone testing.
    """
    
    # Load and download the dataset if not provided
    if malaria_builder is None:
        malaria_builder = load()
        malaria_builder.download_and_prepare()
    
    train, val, test = split(malaria_builder, batch_size)

    # Same preprocessing for all splits (keeps them on the same scale)
    train = train.map(preprocess_image_dataset)
    val = val.map(preprocess_image_dataset)
    test = test.map(preprocess_image_dataset)

    # Augmentation on the training set only
    train = augment(train, batch_size)

    return train, val, test
