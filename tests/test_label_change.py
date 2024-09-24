import os


def test_file_exists():
    assert os.path.exists("labels.txt")
    assert os.path.exists("classification_model.tflite")


def test_file_contains_green_square():
    with open("labels.txt", "r") as file:
        content = file.read()
    assert "green_square" in content
