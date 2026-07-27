# Public API of the model package.
# from anomaly_detection.model import build_cnn
from .architectures import build_cnn

__all__ = ["build_cnn"]
