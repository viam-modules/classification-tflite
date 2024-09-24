import argparse
import json
import os
import typing as ty
import tensorflow as tf
from keras import Model
import numpy as np


# Hello, this is a test. 

single_label = "MODEL_TYPE_SINGLE_LABEL_CLASSIFICATION"
multi_label = "MODEL_TYPE_MULTI_LABEL_CLASSIFICATION"
labels_filename = "labels.txt"
unknown_label = "UNKNOWN"


TFLITE_OPS = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]


def parse_args():
    """Returns dataset file, model output directory, and num_epochs if present. These must be parsed as command line
    arguments and then used as the model input and output, respectively. The number of epochs can be used to optionally override the default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", dest="data_json", type=str)
    parser.add_argument("--model_output_directory", dest="model_dir", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int)
    args = parser.parse_args()
    return args.data_json, args.model_dir, args.num_epochs


def parse_filenames_and_labels_from_json(
    filename: str, all_labels: ty.List[str], model_type: str
) -> ty.Tuple[ty.List[str], ty.List[str]]:
    """Load and parse JSON file to return image filenames and corresponding labels.
       The JSON file contains lines, where each line has the key "image_path" and "classification_annotations".
    Args:
        filename: JSONLines file containing filenames and labels
        all_labels: list of all N_LABELS
        model_type: string single_label or multi_label
    """
    image_filenames = []
    image_labels = []

    with open(filename, "rb") as f:
        for line in f:
            json_line = json.loads(line)
            image_filenames.append(json_line["image_path"])

            annotations = json_line["classification_annotations"]
            labels = [unknown_label]
            for annotation in annotations:
                if model_type == multi_label:
                    if annotation["annotation_label"] in all_labels:
                        labels.append(annotation["annotation_label"])
                # For single label model, we want at most one label.
                # If multiple valid labels are present, we arbitrarily select the last one.
                if model_type == single_label:
                    if annotation["annotation_label"] in all_labels:
                        labels = [annotation["annotation_label"]]
            image_labels.append(labels)
    return image_filenames, image_labels


def get_neural_network_params(
    num_classes: int, model_type: str
) -> ty.Tuple[str, str, str, str]:
    """Function that returns units and activation used for the last layer
        and loss function for the model, based on number of classes and model type.
    Args:
        num_classes: number of classes to be predicted by the model
        model_type: string single-label or multi-label for desired output
    """
    # Single-label Classification
    if model_type == single_label:
        units = num_classes
        activation = "softmax"
        loss = tf.keras.losses.categorical_crossentropy
        metrics = (
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        )
    # Multi-label Classification
    elif model_type == multi_label:
        units = num_classes
        activation = "sigmoid"
        loss = tf.keras.losses.binary_crossentropy
        metrics = (
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        )
    return units, activation, loss, metrics


def preprocessing_layers_classification(
    img_size: ty.Tuple[int, int] = (256, 256)
) -> ty.Tuple[tf.Tensor, tf.Tensor]:
    """Preprocessing steps to apply to all images passed through the model.
    Args:
        img_size: optional 2D shape of image
    """
    preprocessing = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(
                img_size[0], img_size[1], crop_to_aspect_ratio=False
            ),
        ]
    )
    return preprocessing


def encoded_labels(
    image_labels: ty.List[str], all_labels: ty.List[str], model_type: str
) -> tf.Tensor:
    """Returns a tuple of normalized image array and hot encoded labels array.
    Args:
        image_labels: labels present in image
        all_labels: list of all N_LABELS
        model_type: string single_label or multi_label
    """
    if model_type == single_label:
        encoder = tf.keras.layers.StringLookup(
            vocabulary=all_labels, num_oov_indices=0, output_mode="one_hot"
        )
    elif model_type == multi_label:
        encoder = tf.keras.layers.StringLookup(
            vocabulary=all_labels, num_oov_indices=0, output_mode="multi_hot"
        )
    return encoder(image_labels)


def parse_image_and_encode_labels(
    filename: str,
    labels: ty.List[str],
    all_labels: ty.List[str],
    model_type: str,
    img_size: ty.Tuple[int, int] = (256, 256),
) -> ty.Tuple[tf.Tensor, tf.Tensor]:
    """Returns a tuple of normalized image array and hot encoded labels array.
    Args:
        filename: string representing path to image
        labels: list of up to N_LABELS associated with image
        all_labels: list of all N_LABELS
        model_type: string single_label or multi_label
        img_size: optional 2D shape of image
    """
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_image(
        image_string,
        channels=3,
        expand_animations=False,
        dtype=tf.dtypes.uint8,
    )

    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [img_size[0], img_size[1]])
    # Convert string labels to encoded labels
    labels_encoded = encoded_labels(labels, all_labels, model_type)
    return image_resized, labels_encoded


def create_dataset_classification(
    filenames: ty.List[str],
    labels: ty.List[str],
    all_labels: ty.List[str],
    model_type: str,
    img_size: ty.Tuple[int, int] = (256, 256),
    train_split: float = 0.8,
    batch_size: int = 64,
    shuffle_buffer_size: int = 1024,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    prefetch_buffer_size: int = tf.data.experimental.AUTOTUNE,
) -> ty.Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load and parse dataset from Tensorflow datasets.
    Args:
        filenames: string list of image paths
        labels: list of string lists, where each string list contains up to N_LABEL labels associated with an image
        all_labels: string list of all N_LABELS
        model_type: string single_label or multi_label
        img_size: optional 2D shape of image
        train_split: optional float between 0.0 and 1.0 to specify proportion of images that will be used for training
        batch_size: optional size for number of samples for each training iteration
        shuffle_buffer_size: optional size for buffer that will be filled and randomly sampled from, with replacement
        num_parallel_calls: optional integer representing the number of batches to compute asynchronously in parallel
        prefetch_buffer_size: optional integer representing the number of batches that will be buffered when prefetching
    """
    # Create a first dataset of file paths and labels
    if model_type == single_label:
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    elif model_type == multi_label:
        dataset = tf.data.Dataset.from_tensor_slices(
            (filenames, tf.ragged.constant(labels))
        )
    else:
        return None, None

    # Apply a map to the dataset that converts filenames and text labels
    # to normalized images and encoded labels, respectively.
    def mapping_fnc(x, y):
        return parse_image_and_encode_labels(x, y, all_labels, model_type, img_size)

    # Parse and preprocess observations in parallel
    dataset = dataset.map(mapping_fnc, num_parallel_calls=num_parallel_calls)

    # Shuffle the data for each buffer size
    # Disabling reshuffling ensures items from the training and test set will not get shuffled into each other
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False
    )

    train_size = int(train_split * len(filenames))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # Batch the data for multiple steps
    # If the size of training data is smaller than the batch size,
    # batch the data to expand the dimensions by a length 1 axis.
    # This will ensure that the training data is valid model input
    train_batch_size = batch_size if batch_size < train_size else train_size
    if model_type == single_label:
        train_dataset = train_dataset.batch(train_batch_size)
    else:
        train_dataset = train_dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(train_batch_size)
        )

    # Fetch batches in the background while the model is training.
    train_dataset = train_dataset.prefetch(buffer_size=prefetch_buffer_size)

    return train_dataset, test_dataset


# Build the Keras model
def build_and_compile_classification(
    labels: ty.List[str], model_type: str, input_shape: ty.Tuple[int, int, int]
) -> Model:
    """Builds and compiles a classification model for fine-tuning using EfficientNetB0 and weights from ImageNet.
    Args:
        labels: list of string lists, where each string list contains up to N_LABEL labels associated with an image
        model_type: string single_label or multi_label
        input_shape: 3D shape of input
    """
    units, activation, loss_fnc, metrics = get_neural_network_params(
        len(labels), model_type
    )

    x = tf.keras.Input(input_shape, dtype=tf.uint8)
    # Data processing
    preprocessing = preprocessing_layers_classification(input_shape[:-1])
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    # Get the pre-trained model
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    # Freeze the weights of the base model. This allows to use transfer learning
    # to train only the top layers of the model. Setting the base model to be trainable
    # would allow for all layers, not just the top, to be retrained.
    base_model.trainable = False
    # Add custom layers
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    # Output layer
    classification = tf.keras.layers.Dense(units, activation=activation, name="output")

    y = tf.keras.Sequential(
        [
            preprocessing,
            data_augmentation,
            base_model,
            global_pooling,
            classification,
        ]
    )(x)

    model = tf.keras.Model(x, y)

    model.compile(
        loss=loss_fnc,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[metrics],
    )
    return model


def save_labels(labels: ty.List[str], model_dir: str) -> None:
    """Saves a label.txt of output labels to the specified model directory.
    Args:
        labels: list of string lists, where each string list contains up to N_LABEL labels associated with an image
        model_dir: output directory for model artifacts
    """
    filename = os.path.join(model_dir, labels_filename)
    with open(filename, "w") as f:
        for label in labels[:-1]:
            f.write(label + "\n")
        f.write(labels[-1])


def save_tflite_classification(
    model: Model,
    model_dir: str,
    model_name: str,
    target_shape: ty.Tuple[int, int, int],
) -> None:
    """Save model as a TFLite model.
    Args:
        model: trained model
        model_dir: output directory for model artifacts
        model_name: name of saved model
        target_shape: desired output shape of predictions from model
    """
    # Convert the model to tflite, with batch size 1 so the graph does not have dynamic-sized tensors.
    input = tf.keras.Input(target_shape, batch_size=1, dtype=tf.uint8)
    output = model(input, training=False)
    wrapped_model = tf.keras.Model(inputs=input, outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(wrapped_model)
    converter.target_spec.supported_ops = TFLITE_OPS
    tflite_model = converter.convert()

    filename = os.path.join(model_dir, f"{model_name}.tflite")
    # Writing the model buffer into a file.
    with open(filename, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    # Set up compute device strategy. If GPUs are available, they will be used
    if len(tf.config.list_physical_devices("GPU")) > 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    IMG_SIZE = (256, 256)
    # Batch size, buffer size, epochs can be adjusted according to the training job.
    BATCH_SIZE = 16
    SHUFFLE_BUFFER_SIZE = 32
    AUTOTUNE = (
        tf.data.experimental.AUTOTUNE
    )  # Adapt preprocessing and prefetching dynamically

    # Model constants
    NUM_WORKERS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS

    DATA_JSON, MODEL_DIR, num_epochs = parse_args()
    EPOCHS = 200 if num_epochs is None or 0 else int(num_epochs)

    # Read dataset file, labels should be changed according to the desired model output.
    LABELS = ["orange_triangle", "blue_star"]
    # The model type can be changed based on whether we want the model to output one label per image or multiple labels per image
    model_type = single_label
    image_filenames, image_labels = parse_filenames_and_labels_from_json(
        DATA_JSON, LABELS, model_type
    )
    # Generate 80/20 split for train and test data
    train_dataset, test_dataset = create_dataset_classification(
        filenames=image_filenames,
        labels=image_labels,
        all_labels=LABELS + [unknown_label],
        model_type=model_type,
        img_size=IMG_SIZE,
        train_split=0.8,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        num_parallel_calls=AUTOTUNE,
        prefetch_buffer_size=AUTOTUNE,
    )

    # Build and compile model
    with strategy.scope():
        model = build_and_compile_classification(
            LABELS + [unknown_label], model_type, IMG_SIZE + (3,)
        )

    # Train model on data
    loss_history = model.fit(
        x=train_dataset,
        epochs=EPOCHS,
    )

    # Save labels.txt file
    save_labels(LABELS + [unknown_label], MODEL_DIR)
    # Convert the model to tflite
    save_tflite_classification(
        model, MODEL_DIR, "classification_model", IMG_SIZE + (3,)
    )
