import argparse
import json
import os
import typing as ty
import tensorflow as tf
from keras import Model
import numpy as np

from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils

if __name__ == "__main__":
    file_path = "new_file.txt"

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write content to the file
        file.write("Hello, this is a new text file created using open() function")
        print(f"I wrote to {file_path}")
