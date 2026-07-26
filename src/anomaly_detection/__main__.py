# Entry point for the package. Run with:  python -m anomaly_detection
# (from the src/ folder, or with PYTHONPATH=src)

from .data.load import load
from .data.visualize import visualize
from .data.preprocess import preprocess_image_dataset, preprocess_and_visualize
from .data.split import split
from .data.augment import augment
from .config import VISUALIZE
from .config import BATCH_SIZE


def main():
    # load the dataset info
    malaria_builder = load()
    malaria_builder.download_and_prepare()

    if VISUALIZE:
        # Visualise few image
        visualize(malaria_builder)

        #Pre process image
        preprocess_and_visualize(malaria_builder, 1)

    # Split data
    train, val, test = split(malaria_builder, BATCH_SIZE)


    # Augment training data
    if VISUALIZE:
        # Visualize single image after augmentation
        augment(train.take(1), visualize=True)

    #pre process all data set once
    train = train.map(preprocess_image_dataset)
    val = val.map(preprocess_image_dataset)
    test = test.map(preprocess_image_dataset)

    #augment training data
    train = augment(train, BATCH_SIZE)



    




if __name__ == "__main__":
    main()
