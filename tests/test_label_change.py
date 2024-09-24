import os


def test_label_change():
    with open("labels.txt", "r") as file:
        content = file.read()
    assert "green_square" in content
