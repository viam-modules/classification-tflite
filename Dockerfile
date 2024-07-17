# Install the 2.11 version of tensorflow
FROM tensorflow/tensorflow:2.11.1-gpu

# Install the dependencies
RUN apt-get update && apt-get install libusb-1.0-0

# Copies the trainer code to the Docker image and builds the source distribution.
COPY . /root/
WORKDIR /root/
RUN pip install poetry
RUN poetry install
RUN poetry run pip install tflite-support
# Install the source distribution


# Set up the entry point to invoke the trainer.
ENTRYPOINT ["poetry", "run", "python3", "model/training.py"]