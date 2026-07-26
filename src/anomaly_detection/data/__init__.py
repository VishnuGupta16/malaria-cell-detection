# Public API of the data package.
# Exposing prepare_data here lets callers write:
# from anomaly_detection.data import prepare_data
from .pipeline import prepare_data

__all__ = ["prepare_data"]
