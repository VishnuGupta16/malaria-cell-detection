from .data import prepare_data
from .config import BATCH_SIZE


def main():

    # Prepare the data: split -> preprocess -> augment (train only)
    train, val, test = prepare_data(BATCH_SIZE)

    # Next phase: build and train the model using train / val / test


if __name__ == "__main__":
    main()
