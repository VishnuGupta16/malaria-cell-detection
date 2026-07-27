import tensorflow as tf
from tensorflow.keras import layers, models
from ..config import MODEL_INPUT_SHAPE


def build_cnn(input_shape=MODEL_INPUT_SHAPE):
    """Build a simple CNN for binary malaria classification.

    Flow:
      Input -> [Conv2D -> MaxPooling2D] x3 -> Flatten -> Dense -> Dense(1, sigmoid)

    Returns an *uncompiled* Keras Sequential model.
    """
    
    

