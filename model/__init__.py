# Configure TensorFlow to use legacy Keras (tf_keras) instead of Keras 3.x
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
