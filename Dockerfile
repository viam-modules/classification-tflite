# Install the 2.11 version of tensorflow
FROM tensorflow/tensorflow:2.11.1-gpu

# Install the dependencies
RUN apt-get update && apt-get install libusb-1.0-0

# Copies the trainer code to the Docker image and builds the source distribution.
COPY . /root/
WORKDIR /root/
RUN python3 setup.py sdist --formats=gztar
# Install the source distribution
RUN pip install dist/model-0.1.tar.gz

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-m", "model.training"]