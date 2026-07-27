from .data import prepare_data
from .config import BATCH_SIZE

from .validate_input import validate_sample



def main():

    # Prepare the data: split -> preprocess -> augment (train only)
    train, val, test = prepare_data(BATCH_SIZE)

    # Validate sample
    result = validate_sample(train_set=train, validation_batch_size=100)
    print(f'{result}')

    # Next phase: build and train the model using train / val / test


if __name__ == "__main__":
    main()
