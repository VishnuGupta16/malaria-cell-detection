# Run ONLY the data pipeline, for individual testing:
#   PYTHONPATH=src python3 -m anomaly_detection.data

from .load import load
from . import prepare_data
from ..config import BATCH_SIZE
from .visualize import visualize
from .preprocess import preprocess_and_visualize
from .augment import augment
from ..config import VISUALIZE

def main():
    malaria_builder = load()
    malaria_builder.download_and_prepare()

     # Optional visual exploration (previews run on raw images)
    if VISUALIZE:
        visualize(malaria_builder)
        preprocess_and_visualize(malaria_builder, 1)
        raw_dataset = malaria_builder.as_dataset(split="train")
        augment(raw_dataset.take(1), visualize=True)

    train, val, test = prepare_data(BATCH_SIZE, malaria_builder=malaria_builder)

    print("Data pipeline complete. Prepared splits:")
    print("  train:", len(train))
    print("  val:  ", len(val))
    print("  test: ", len(test))


if __name__ == "__main__":
    main()
