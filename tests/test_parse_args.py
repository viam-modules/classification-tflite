import model.training


def test_parse_args():
    args = [
        "--dataset_file=test_data.json",
        "--model_output_directory=test_output/",
        "--num_epochs=5",
        "--labels='label_1 label_2'",
        "--model_type=multi_label",
    ]

    data_json, model_dir, num_epochs, labels, model_type = model.training.parse_args(
        args
    )

    assert (
        data_json == "test_data.json"
    ), f"Expected dataset_file to be 'test_data.json', got {data_json}"
    assert (
        model_dir == "test_output/"
    ), f"Expected model_output_directory to be 'test_output/', got {model_dir}"
    assert num_epochs == 5, f"Expected num_epochs to be 5, got {num_epochs}"
    assert (
        labels == "'label_1 label_2'"
    ), f"Expected labels to be 'label_1 label_2', got {labels}"
    assert (
        model_type == "multi_label"
    ), f"Expected model_type to be multi_label, got {model_type}"
