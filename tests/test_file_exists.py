import os


def test_file_exists():
    print("here i am")
    assert os.path.exists("new_file.txt")
