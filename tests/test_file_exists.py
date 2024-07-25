import os


def test_file_exists():
    assert os.path.exists("labels.txt")
    assert os.path.exists("classification_model.tflite")
