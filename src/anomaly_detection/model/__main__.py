# Build the CNN and print its architecture:
#   PYTHONPATH=src python3 -m anomaly_detection.model
from .architectures import build_cnn


def main():
    model = build_cnn()
    model.summary()


if __name__ == "__main__":
    main()
