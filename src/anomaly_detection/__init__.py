import os

# Suppress INFO & WARNING logs from TensorFlow.
# This project's logs often mention GPU drivers or Google Cloud authentication,
# even if you are only running on CPU. They are safe to ignore in this project.
# This must run before TensorFlow is imported anywhere, so it lives in __init__.py,
# which Python runs first when you start the package.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
