import tensorflow as tf

import model.training

ALL_LABELS = ["label_1", "label_2", model.training.unknown_label]


def test_encoded_labels_single_label():
    encoded = model.training.encoded_labels(
        tf.constant(["label_2"]), ALL_LABELS, model.training.single_label
    )
    assert encoded.shape == (3,), f"Expected shape (3,), got {encoded.shape}"
    assert encoded.numpy().tolist() == [0, 1, 0]


def test_encoded_labels_multi_label():
    encoded = model.training.encoded_labels(
        tf.constant([model.training.unknown_label, "label_1", "label_2"]),
        ALL_LABELS,
        model.training.multi_label,
    )
    assert encoded.shape == (3,), f"Expected shape (3,), got {encoded.shape}"
    assert encoded.numpy().tolist() == [1, 1, 1]
