import os


def test_file_exists():
    assert os.path.exists("new_file.txt") == False
